import logging

import numpy
from beartype.typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
from einops import reduce, rearrange, repeat

from .modules import exists, default
from .pos_emb import LearnedSinusoidalPosEmb
from .convnext import ConvNeXtBlock
from .transformer import Transformer, ConvPositionEmbed
from .melvoco import MelVoco


# tensor helpers
def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob


def mask_for_freqency(cond, batch: int, seq_len: int, mel_dim: int, device):

    for i in range(batch):

        import random
        mask_height = random.randint(10,20)
        rand_start = random.randint(20, mel_dim - mask_height)
        minimum = torch.min(cond)
        cond[i, :, rand_start: rand_start + mask_height] = minimum + 1e-3

    return cond


def reduce_masks_with_and(*masks):
    masks = [*filter(exists, masks)]

    if len(masks) == 0:
        return None

    mask, *rest_masks = masks

    for rest_mask in rest_masks:
        mask = mask & rest_mask

    return mask


class FLowHigh(nn.Module):
    def __init__(
        self,
        *,
        audio_enc_dec: Optional[MelVoco] = None,
        dim_in = None, # 256
        dim_cond_emb = 0,
        dim = 1024,
        depth = 24,
        dim_head = 64,
        heads = 16,
        ff_mult = 4,
        ff_dropout = 0.,
        time_hidden_dim = None,
        conv_pos_embed_kernel_size = 31,
        conv_pos_embed_groups = None,
        attn_dropout = 0.,
        attn_flash = False,
        attn_qk_norm = True,
        use_gateloop_layers = False,
        architecture = "transformer",
    ):
        super().__init__()
        dim_in = default(dim_in, dim)

        self.architecture = architecture

        if self.architecture=='transformer':
            time_hidden_dim = default(time_hidden_dim, dim)
        elif self.architecture=='convnext':
            time_hidden_dim = default(time_hidden_dim, dim)
        else:
            raise ValueError("Choose approriate architecture")

        self.audio_enc_dec = audio_enc_dec

        self.proj_in = nn.Identity()

        self.sinu_pos_emb = nn.Sequential(
            LearnedSinusoidalPosEmb(dim),
            nn.Linear(dim, time_hidden_dim),
            nn.SiLU()
        )

        self.dim_cond_emb = dim_cond_emb
        self.to_embed = nn.Linear(dim_in * 2 + dim_cond_emb, dim)
        self.null_cond = nn.Parameter(torch.zeros(dim_in), requires_grad = False)
        self.conv_embed = ConvPositionEmbed(
            dim = dim,
            kernel_size = conv_pos_embed_kernel_size,
            groups = conv_pos_embed_groups
        )

        if self.architecture =='transformer':

            self.transformer = Transformer(
                dim = dim,
                depth = depth,
                dim_head = dim_head,
                heads = heads,
                ff_mult = ff_mult,
                ff_dropout = ff_dropout,
                attn_dropout= attn_dropout,
                attn_flash = attn_flash,
                attn_qk_norm = attn_qk_norm,
                adaptive_rmsnorm = True,
                adaptive_rmsnorm_cond_dim_in = time_hidden_dim,
                use_gateloop_layers = use_gateloop_layers
            )

        elif self.architecture=='convnext':
            intermediate_dim = dim * 3
            num_layers = 8
            layer_scale_init_value = 1
            self.convnext = nn.ModuleList(
                [
                    ConvNeXtBlock(
                        dim=dim,
                        intermediate_dim=intermediate_dim,
                        layer_scale_init_value=layer_scale_init_value,
                        hidden_dim=time_hidden_dim,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)

        dim_out = dim_in
        self.to_pred = nn.Linear(dim, dim_out, bias = False)

    @property
    def device(self):
        return next(self.parameters()).device

    def hz_to_mel(self,f):
        if isinstance(f, (list, numpy.ndarray)):
            f = numpy.array(f)
        return 2595 * numpy.log10(1 + f / 700)

    def mel_bin_index(self, frequency, sample_rate, num_mel_bins):
        nyquist = sample_rate / 2
        m_min = self.hz_to_mel(0)
        m_max = self.hz_to_mel(nyquist)
        mel_value = self.hz_to_mel(frequency)
        bin_index = numpy.floor((mel_value - m_min) / (m_max - m_min) * num_mel_bins)
        if isinstance(bin_index, numpy.ndarray):
            bin_index = bin_index.astype(int)
        else:
            bin_index = int(bin_index)
        return bin_index

    @torch.inference_mode()
    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1.:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        *,
        times,
        self_attn_mask = None,
        cond_drop_prob = 0.1,
        target = None,
        cond = None,
        cond_mask = None,
        cond_freq_masking = False,
        random_sr = None,
        weighted_loss = False,
        cutoff_bins = None,
    ):

        x = self.proj_in(x)
        cond = default(cond, target)

        if exists(cond):
            cond = self.proj_in(cond)

        # shapes
        batch, seq_len, cond_dim = cond.shape
        assert cond_dim == x.shape[-1]


        # auto manage shape of times, for odeint times
        if times.ndim == 0:
            times = repeat(times, '-> b', b = cond.shape[0])
        if times.ndim == 1 and times.shape[0] == 1:
            times = repeat(times, '1 -> b', b = cond.shape[0])

        # Cond frequency masking
        if cond_freq_masking:
            if self.training:
                cond = mask_for_freqency(cond, batch, seq_len, cond_dim, device=self.device)
            else:
                cond_freq_mask = torch.ones((batch, seq_len,cond_dim), device = cond.device, dtype =torch.bool)
                cond = cond * cond_freq_mask
        else:
            pass

        # Classifier free guidance
        if cond_drop_prob > 0.:
            cond_drop_mask = prob_mask_like(cond.shape[:1], cond_drop_prob, self.device)
            cond = torch.where(
                rearrange(cond_drop_mask, '... -> ... 1 1'),
                self.null_cond,
                cond
            )

        # x.shape : [B, Time, channel]
        # cond.shape : [B, Time, channel]
        to_concat = [*filter(exists, (x,cond))]

        # embed.shape : [B, Time, dim_in*2 ]
        embed = torch.cat(to_concat, dim = -1)

        x = self.to_embed(embed)
        x = self.conv_embed(x, mask = self_attn_mask) + x

        time_emb = self.sinu_pos_emb(times)

        if self.architecture=='transformer':
            x = self.transformer(x, mask = self_attn_mask, adaptive_rmsnorm_cond = time_emb)

        elif self.architecture=='convnext':
            x = x.transpose(1,2)
            for convnext_block in self.convnext:
                x = convnext_block(x, cond=time_emb)

            x = x.transpose(1,2)
            x = self.final_layer_norm(x)

        # Protect NaN
        logging.info(f"After transformer: {x}")
        if torch.isnan(x).any():
            print(x)
            logging.error("NaN detected after main architecture")

        x = self.to_pred(x)

        # Protect NaN
        logging.info(f"After predict: {x}")
        if torch.isnan(x).any():
            print(x)
            logging.error("NaN detected after last projection layer")


        # if no target passed in, just return logits
        # for inference mode
        if not exists(target):

            return x

        loss_mask = reduce_masks_with_and(cond_mask, self_attn_mask)

        if not exists(loss_mask):

            if weighted_loss == False:
                return F.mse_loss(x, target)

            elif weighted_loss == True:
                low_weight = 1.0
                high_weight = 2.0
                n_mels = self.audio_enc_dec.n_mels
                weight = torch.ones(batch,n_mels) * low_weight
                if isinstance(cutoff_bins, numpy.ndarray):
                    for i, bin_idx in enumerate(cutoff_bins):
                        weight[i, bin_idx:] = high_weight
                else:
                    exit()
                    weight[cutoff_bins:] = high_weight

                weight = weight.unsqueeze(1).expand(batch, seq_len, n_mels).cuda()
                mse_loss = F.mse_loss(x, target, reduction='none')
                weighted_mse_loss = mse_loss * weight
                mean_loss = weighted_mse_loss.mean()
                return mean_loss

        loss = F.mse_loss(x, target, reduction = 'none')
        loss = reduce(loss, 'b n d -> b n', 'mean')
        loss = loss.masked_fill(~loss_mask, 0.)

        # masked mean
        num = reduce(loss, 'b n -> b', 'sum')
        den = loss_mask.sum(dim = -1).clamp(min = 1e-5)
        loss = num / den
        return loss.mean()
