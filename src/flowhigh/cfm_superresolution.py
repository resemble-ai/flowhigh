import logging
from random import random
from functools import partial
from pathlib import Path

from beartype import beartype

from einops import rearrange, repeat, reduce, pack, unpack
import numpy
from librosa.filters import mel as librosa_mel_fn
import torch
from torch import nn, Tensor
from torch.nn import Module
import torch.nn.functional as F
import torchode as to
from torchdiffeq import odeint
import torchaudio.transforms as T
from torchaudio.functional import resample

from .utils import sequence_mask

from postprocessing import PostProcessing
from models.modules import exists, default
from models import FLowHigh


LOGGER = logging.getLogger(__file__)
logging.basicConfig(filename='model_debug.log', level=logging.INFO)


# helper functions

def divisible_by(num, den):
    return (num % den) == 0

def is_odd(n):
    return not divisible_by(n, 2)

def coin_flip():
    return random() < 0.5

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# mel helpers
mel_basis = {}
hann_window = {}



def interpolate_1d(t, length, mode = 'bilinear'):
    " pytorch does not offer interpolation 1d, so hack by converting to 2d "

    dtype = t.dtype
    t = t.float()

    implicit_one_channel = t.ndim == 2
    if implicit_one_channel:
        t = rearrange(t, 'b n -> b 1 n')

    t = rearrange(t, 'b d n -> b d n 1')
    t = F.interpolate(t, (length, 1), mode = mode)
    t = rearrange(t, 'b d n 1 -> b d n')

    if implicit_one_channel:
        t = rearrange(t, 'b 1 n -> b n')

    t = t.to(dtype)

    return t

def curtail_or_pad(t, target_length):
    length = t.shape[-2]

    if length > target_length:
        t = t[..., :target_length, :]
    elif length < target_length:
        t = F.pad(t, (0, 0, 0, target_length - length), value = 0.)

    return t

# mask construction helpers
def mask_from_start_end_indices(seq_len: int, start: Tensor, end: Tensor):
    assert start.shape == end.shape
    device = start.device

    seq = torch.arange(seq_len, device = device, dtype = torch.long)
    seq = seq.reshape(*((-1,) * start.ndim), seq_len)
    seq = seq.expand(*start.shape, seq_len)

    mask = seq >= start[..., None].long() # start
    mask &= seq < end[..., None].long()

    return mask

def mask_from_frac_lengths(seq_len: int, frac_lengths: Tensor):
    device = frac_lengths.device

    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.zeros_like(frac_lengths, device = device).float().uniform_(0, 1)
    start = (max_start * rand).clamp(min = 0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)

def is_probably_audio_from_shape(t):
    return exists(t) and (t.ndim == 2 or (t.ndim == 3 and t.shape[1] == 1))

class ConditionalFlowMatcherWrapper(Module):
    @beartype
    def __init__(
        self,
        flowhigh: FLowHigh,
        sigma = 0.,
        ode_atol = 1e-5,
        ode_rtol = 1e-5,
        use_torchode = False,
        cfm_method = 'basic_cfm',
        torchdiffeq_ode_method = 'midpoint',   # [euler, midpoint]
        torchode_method_klass = to.Tsit5,
        cond_drop_prob = 0.
    ):
        super().__init__()
        self.sigma = sigma
        self.flowhigh = flowhigh
        self.cond_drop_prob = cond_drop_prob
        self.use_torchode = use_torchode
        self.torchode_method_klass = torchode_method_klass
        self.cfm_method = cfm_method
        self.odeint_kwargs = dict(
            atol = ode_atol,
            rtol = ode_rtol,
            method = torchdiffeq_ode_method
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def load(self, path, strict = True):
        # return pkg so the trainer can access it
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = 'cpu')
        self.load_state_dict(pkg['model'], strict = strict)
        return pkg

    # For mel repalcement
    def locate_cutoff_freq(self, mel, percentile=0.9995):
        def find_cutoff(x, percentile=0.99):
            percentile = x[-1] * percentile
            for i in range(1, x.shape[0]):
                if x[-i] < percentile:
                    return x.shape[0] - i
            return 0

        magnitude = torch.abs(mel)
        energy = torch.cumsum(torch.sum(magnitude, dim=0), dim=0)
        return find_cutoff(energy, percentile)

    def mel_replace_ops(self, samples, input, cutoff_melbins):
        result = torch.zeros_like(samples)
        for i in range(samples.size(0)):

            result[i][..., cutoff_melbins[i]:] = samples[i][..., cutoff_melbins[i]:]
            result[i][..., :cutoff_melbins[i]] = input[i][..., :cutoff_melbins[i]]
        return result, cutoff_melbins

    def mel_cutoff_bins(self, input):
        cutoff_melbins = []
        for i in range(input.size(0)):
            cutoff_melbin = self.locate_cutoff_freq(torch.exp(input[i]))
            cutoff_melbins.append(cutoff_melbin)
        return cutoff_melbins


    @torch.inference_mode()
    def sample(
        self,
        *,
        cond = None,
        cond_mask = None,
        time_steps = 4,
        cond_scale = 1.,
        decode_to_audio = True,
        std_1 = None,
        std_2 = None,
        mel_pp = False,
        cfm_method = None
    ):
        if cfm_method not in ['basic_cfm','independent_cfm_adaptive', 'independent_cfm_constant', 'independent_cfm_mix']:
            cfm_method = self.cfm_method
            # raise ValueError("Do not define cfm_method variable for sample()")

        if cfm_method in ['independent_cfm_adaptive', 'independent_cfm_constant','independent_cfm_mix']:
            if std_1 is None or std_2 is None:
                std_1 = 1.0
                std_2 = self.sigma

        cond_is_raw_audio = is_probably_audio_from_shape(cond)

        if cond_is_raw_audio:
            assert exists(self.flowhigh.audio_enc_dec)

            self.flowhigh.audio_enc_dec.eval()
            cond = self.flowhigh.audio_enc_dec.encode(cond)

        self_attn_mask = None
        shape = cond.shape # [B, Time, Channel]
        batch = shape[0]
        cutoff_bins = self.mel_cutoff_bins(cond)

        # neural ode
        self.flowhigh.eval()

        # ode function
        def ode_fn(t, x, *, packed_shape = None):
            if exists(packed_shape):
                x = unpack_one(x, packed_shape, 'b *')

            out = self.flowhigh.forward_with_cond_scale(
                x,
                times = t,
                cond = cond,
                cond_scale = cond_scale,
                cond_mask = cond_mask,
                self_attn_mask = self_attn_mask
            )

            if exists(packed_shape):
                out = rearrange(out, 'b ... -> b (...)')
            return out # out.shape : [1, Time, mel_channel]

        if cfm_method == 'basic_cfm':
            y0 = torch.randn_like(cond).cuda()

        elif cfm_method == 'independent_cfm_adaptive':
            # y0 from intended prior
            epsilon = torch.randn_like(cond).cuda()
            y0 = cond*std_1 + epsilon*std_2

        elif cfm_method == 'independent_cfm_constant':
            # y0 from intended prior
            epsilon = torch.randn_like(cond).cuda()
            y0 = cond*std_1 + epsilon*std_2

        elif cfm_method == 'independent_cfm_mix':
            # y0 from intended prior
            epsilon = torch.randn_like(cond).cuda()
            y0_low = cond*std_1 + epsilon*std_2
            y0_high = epsilon
            y0, _ = self.mel_replace_ops(y0_high, y0_low, cutoff_bins)

        t = torch.linspace(0, 1, time_steps + 1, device = self.device).cuda()
        if not self.use_torchode:

            LOGGER.debug('sampling with torchdiffeq')
            trajectory = odeint(ode_fn, y0, t, **self.odeint_kwargs) # bottle neck
            sampled = trajectory[-1]

            # # trajectory plot
            # n = len(trajectory)
            # for i in range(n):
            #     plt.figure(figsize=(12, 4))
            #     plt.imshow(numpy.rot90(trajectory[i].squeeze().cpu().numpy(), 1), aspect='auto', origin='upper', interpolation='none')
            #     plt.colorbar()
            #     plt.title(f'trajectory[{i}]')
            #     plt.xlabel('X-axis')
            #     plt.ylabel('Y-axis')

            #     plt.savefig(f'__trajectory[{i}].png', dpi=300, bbox_inches='tight')
            #     plt.close()

        else:
            LOGGER.debug('sampling with torchode')
            t = repeat(t, 'n -> b n', b = batch)
            y0, packed_shape = pack_one(y0, 'b *')
            fn = partial(ode_fn, packed_shape = packed_shape)
            term = to.ODETerm(fn)
            step_method = self.torchode_method_klass(term = term)
            step_size_controller = to.IntegralController(
                atol = self.odeint_kwargs['atol'],
                rtol = self.odeint_kwargs['rtol'],
                term = term
            )
            solver = to.AutoDiffAdjoint(step_method, step_size_controller)
            jit_solver = torch.compile(solver)
            init_value = to.InitialValueProblem(y0 = y0, t_eval = t)
            sol = jit_solver.solve(init_value)
            sampled = sol.ys[:, -1]
            sampled = unpack_one(sampled, packed_shape, 'b *')

        if mel_pp:
            sampled, cutoff_bins = self.mel_replace_ops(sampled, cond, cutoff_bins)

        if not decode_to_audio or not exists(self.flowhigh.audio_enc_dec):
            return sampled

        return self.flowhigh.audio_enc_dec.decode(sampled)

    # this is for training only.
    def forward(
        self,
        x1,
        *,
        mask = None,
        cond = None,
        cond_mask = None,
        cond_lengths = None,
        input_sampling_rate = None,
        cond_freq_masking = False,
        random_sr = None,
        weighted_loss = None,
        cfm_method = None # not necessary
    ):

        if cfm_method not in ['basic_cfm','independent_cfm_adaptive' ,'independent_cfm_constant','independent_cfm_mix']:
            cfm_method = self.cfm_method

        batch, seq_len, dtype, sigma_min = *x1.shape[:2], x1.dtype, self.sigma
        input_is_raw_audio, cond_is_raw_audio = map(is_probably_audio_from_shape, (x1, cond))

        if any([input_is_raw_audio, cond_is_raw_audio]):
            assert exists(self.flowhigh.audio_enc_dec), 'audio_enc_dec must be set on FLowHigh to train directly on raw audio'
            audio_enc_dec_sampling_rate = self.flowhigh.audio_enc_dec.sampling_rate
            input_sampling_rate = default(input_sampling_rate, audio_enc_dec_sampling_rate)

            with torch.no_grad():
                self.flowhigh.audio_enc_dec.eval()
                # Making Ground truth mel-spectrogram
                if input_is_raw_audio:
                    x1 = resample(x1, input_sampling_rate, audio_enc_dec_sampling_rate)
                    x1 = self.flowhigh.audio_enc_dec.encode(x1) # x1.shape : [B, Time, channel]

                # Making mel-spectrogram which are empty in high-freqeuncy information
                if exists(cond) and cond_is_raw_audio:
                    cond = resample(cond, input_sampling_rate, audio_enc_dec_sampling_rate)
                cond = self.flowhigh.audio_enc_dec.encode(cond) # cond.shape : [B, Time, channel]

        if x1.size(1) != cond.size(1):
            max_timelength = max(x1.size(1), cond.size(1))
            x1 = F.pad(x1, (0, 0,max_timelength - x1.size(1), 0 ))
            cond = F.pad(cond, (0, 0, max_timelength - cond.size(1), 0))

        # main conditional flow logic is below
        times = torch.rand((batch,), dtype = dtype, device = self.device)
        t = rearrange(times, 'b -> b 1 1')

        if cfm_method == 'basic_cfm':
            """
            probability path: N(t x1, 1 - (1 - sigma) t)
            mu_t: t * x1
            sigma_t: 1 - (1 - sigma_min)t
            sigma_min = 1e-4

            sample x_t: sigma_t * x0 + t * x1 = (1 - (1 - sigma_min) * t) * x0 + t * x1
            target vector field: u_t = (x1 - (1 - sigma_min) x_t) / (1 - (1 - sigma_min) t) = x1 - (1 - sigma_min) * x0

            if sigma_min = 0, then basic_cfm same with rectified-flow from standard normal distribution N(0,I)
            """
            # x0 is gaussian noise
            x0 = torch.randn_like(x1)   # [B, Time, channel]
            cutoff_bins = None

            sigma_t = (1 - (1 - sigma_min) * t)

            # sample xt = noisy speech (\psi_t (x_0|x_1))
            w = sigma_t * x0 + t * x1  # [B, Time, channel]
            # w = (1 - (1 - sigma_min) * t) * x0 + t * x1  # [B, Time, channel]

            # target vector field u_t
            flow = x1 - (1 - sigma_min) * x0  # [B, Time, channel]
            # flow = (x1 - (1 - sigma_min) * w) / (1- (1 - sigma_min) *t)  # [B, Time, channel]

        elif cfm_method == 'independent_cfm_adaptive':
            """
            q(z) = q(x0)q(x1)
            probability path: N(t * x1 + (1 - t) *x0, 1 - (1 - sigma_min) t)
            mu_t: t * x1 + (1 - t) *x0
            sigma_t:  1 - (1 - sigma_min) t

            sample x_t: mean + sigma * eps = t * x1 + (1 - t) *x0 + sigma_t * epsilon
            target vector field: u_t = { (x1-x0) - (1-sigma_min)(xt-x0) } / { 1 - (1 - sigma_min) t } = (x1-x0) - (1-sigma_min) * epsilon

            if sigma_min = 0, then independent_cfm same with rectified-flow from arbitrary distribution q(x0)
            """

            # eps ~ N(0,I)
            epsilon = torch.randn_like(cond)

            # x0 represents low resolution audio(mel-spectogram)
            x0 = cond.detach().clone()

            cutoff_bins = None

            mu_t = t * x1 + (1 - t) * x0
            sigma_t = (1 - (1 - sigma_min)*t)

            # sample xt
            w = mu_t + sigma_t * epsilon

            # target vector field u_t
            flow = (x1-x0) - (1-sigma_min) * epsilon # { (x1-x0) - (1-sigma_min)*(w-x0) } / {1 - (1 - sigma_min)*t} # [B, Time, channel]


        elif cfm_method == 'independent_cfm_constant':
            """
            q(z) = q(x0)q(x1)
            probability path: N(t * x1 + (1 - t) *x0, sigma_t)
            mu_t: t * x1 + (1 - t) *x0
            sigma_t: sigma_min (small enough)

            sample x_t: mean + sigma*eps(eps~N(0,I)) = t * x1 + (1 - t) *x0 + sigma_t * epsilon
            target vector field: u_t = x1 - x0

            if sigma_min = 0, then independent_cfm same with rectified-flow from arbitrary distribution q(x0)
            """

            # eps ~ N(0,I)
            epsilon = torch.randn_like(cond)

            cutoff_bins = None

            # x0 represents low resolution audio(mel-spectogram)
            x0 = cond.detach().clone()

            mu_t = t * x1 + (1 - t) * x0
            sigma_t = sigma_min

            # sample xt
            w = mu_t + sigma_t * epsilon

            # target vector field u_t
            flow = x1 - x0  # [B, Time, channel]


        elif cfm_method == 'independent_cfm_mix':
            """
            q(z) = q(x0)q(x1)
            probability path_high: N(    t * x1          , 1 - (1 - sigma) t)
            probability path_low : N(t * x1 + (1 - t) *x0,       sigma_min    )

            x0: x^mel_low

            sample x_t:
            target vector field: u_t =
            """

            # # eps ~ N(0,I)
            epsilon = torch.randn_like(cond)

            # get cutoff mel bins of LR mel
            cutoff_bins = self.mel_cutoff_bins(cond)

            # x0 represents low resolution audio(mel-spectogram)
            x0 = cond.detach().clone()

            # sample xt_high
            mu_t_high = t * x1
            sigma_t_high = (1 - (1 - sigma_min) * t)
            xt_high = mu_t_high + sigma_t_high * epsilon   # [B, Time, channel]

            # sample xt_low
            mu_t_low = t * x1 + (1 - t) * x0
            sigma_t_low = sigma_min
            xt_low = mu_t_low + sigma_t_low * epsilon

            w, _ = self.mel_replace_ops(xt_high, xt_low, cutoff_bins)

            # target vector field u_t
            flow = torch.zeros_like(x1)
            flow_high = x1 - (1 - sigma_min) * epsilon
            flow_low = x1 - x0  # [B, Time, channel]＼
            for i, cutoff_bin in enumerate(cutoff_bins):
                flow[i][..., cutoff_bin:] = flow_high[i][..., cutoff_bin:]
                flow[i][..., :cutoff_bin] = flow_low[i][..., :cutoff_bin]

        # x1.shape = cond.shape = x0.shape = w.shape = flow.shape = [Batch, Time, mel_bin]

        # Training mode!
        self.flowhigh.train()

        # Cut a small segment of mel-spectrogram
        cond_lengths = cond_lengths.to(torch.int32)
        max_cond_lengths = x1.size(1)
        x_mask = sequence_mask(cond_lengths, max_cond_lengths).unsqueeze(1)
        out_size = 2* self.flowhigh.audio_enc_dec.sampling_rate // self.flowhigh.audio_enc_dec.hop_length

        # Cut a small segment of mel-spectrogram in order to increase batch size
        if not isinstance(out_size, type(None)):
            max_offset = (cond_lengths - out_size).clamp(0)
            offset_ranges = list( zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))

            import random

            out_offset = torch.LongTensor(
                [
                    torch.tensor(random.choice(range(start, end)) if end > start else 0)
                    for start, end in offset_ranges
                ]
            ).to(cond_lengths)

            w_cut = torch.zeros(w.shape[0], out_size, self.flowhigh.audio_enc_dec.n_mels, dtype=w.dtype, device=w.device)
            flow_cut = torch.zeros(flow.shape[0], out_size, self.flowhigh.audio_enc_dec.n_mels, dtype=flow.dtype, device=flow.device)
            cond_cut = torch.zeros(cond.shape[0], out_size, self.flowhigh.audio_enc_dec.n_mels, dtype=cond.dtype, device=cond.device)

            x_cut_lengths = []

            for i, (w_, flow_, cond_, out_offset_) in enumerate(zip(w, flow, cond, out_offset)):

                # w_.shape = flow_.shape = cond_.shape = [Time, channel]
                x_cut_length = out_size + (cond_lengths[i] - out_size).clamp(None, 0)
                x_cut_lengths.append(x_cut_length)

                cut_lower, cut_upper = out_offset_, out_offset_ + x_cut_length
                w_cut[i, :x_cut_length,: ] = w_[cut_lower:cut_upper,: ]
                flow_cut[i, :x_cut_length,: ] = flow_[cut_lower:cut_upper,: ]
                cond_cut[i, :x_cut_length,: ] = cond_[cut_lower:cut_upper,: ]

            x_cut_lengths = torch.LongTensor(x_cut_lengths)
            x_cut_mask = sequence_mask(x_cut_lengths).unsqueeze(1).to(x_mask)

            w = w_cut
            flow = flow_cut
            cond = cond_cut
            x_mask = x_cut_mask

        # forward
        loss = self.flowhigh(
            x = w,
            cond = cond,
            cond_mask = cond_mask,
            times = times,
            target = flow,
            self_attn_mask = mask,
            cond_drop_prob = self.cond_drop_prob,
            cond_freq_masking = cond_freq_masking,
            random_sr = random_sr,
            weighted_loss = weighted_loss,
            cutoff_bins = cutoff_bins
        )
        return loss

