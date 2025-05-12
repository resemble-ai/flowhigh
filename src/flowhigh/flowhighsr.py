import logging
from random import random
from pathlib import Path

import scipy
import numpy as np
import librosa
import torch
import torch.nn.functional as F
import torchode

from .models import FLowHigh, MelVoco
from .cfm_superresolution import ConditionalFlowMatcherWrapper
from .postprocessing import PostProcessing


class FlowHighSR(ConditionalFlowMatcherWrapper):
    def __init__(
        self,
        flowhigh: FLowHigh,
        sigma = 0.,
        ode_atol = 1e-5,
        ode_rtol = 1e-5,
        use_torchode = False,
        cfm_method = 'basic_cfm',
        torchdiffeq_ode_method = 'midpoint',   # [euler, midpoint]
        torchode_method_klass = torchode.Tsit5,
        cond_drop_prob = 0.,
        #
        upsampling_method='scipy',
    ):
        super().__init__(
            flowhigh=flowhigh,
            sigma=sigma,
            ode_atol=ode_atol,
            ode_rtol=ode_rtol,
            use_torchode=use_torchode,
            cfm_method=cfm_method,
            torchdiffeq_ode_method=torchdiffeq_ode_method,
            torchode_method_klass=torchode_method_klass,
            cond_drop_prob=cond_drop_prob,
        )
        self.upsampling_method = upsampling_method
        self.postproc = PostProcessing(0)
        # self.device = device

    @torch.no_grad()
    def generate(
        self,
        audio: np.ndarray,
        sr: int,
        target_sampling_rate=48000,
        timestep=1,
    ):
        if len(audio.shape) == 2:
            audio = audio.squeeze(0)

        if audio.max() > 1:
            audio = audio / 32768.0

        # Up sampling the input audio (in Numpy)
        if self.upsampling_method =='scipy':
            # audio, sr = librosa.load(wav_file, sr=None, mono=True)
            cond = scipy.signal.resample_poly(audio, target_sampling_rate, sr)
            cond /= np.max(np.abs(cond))
            if isinstance(cond, np.ndarray):
                cond = torch.tensor(cond).unsqueeze(0)
            cond = cond.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')) # [1, T]

        elif self.upsampling_method == 'librosa':
            # audio, sr = librosa.load(wav_file, sr=None, mono=True)
            cond = librosa.resample(audio, sr, target_sampling_rate, res_type='soxr_hq')
            cond /= np.max(np.abs(cond))
            if isinstance(cond, np.ndarray):
                cond = torch.tensor(cond).unsqueeze(0)
            cond = cond.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')) # [1, T]

        # Audio must be in torch.Tensor from now on
        if isinstance(cond, np.ndarray):
            cond = torch.from_numpy(cond)

        cond = cond.float().to(self.device)

        # reconstruct high resolution sample
        if self.cfm_method == 'basic_cfm':
            HR_audio = self.sample(cond = cond, time_steps = timestep, cfm_method = self.cfm_method)
        elif self.cfm_method == 'independent_cfm_adaptive':
            HR_audio = self.sample(cond = cond, time_steps = timestep, cfm_method = self.cfm_method, std_2 = 1.)
        elif self.cfm_method == 'independent_cfm_constant':
            HR_audio = self.sample(cond = cond, time_steps = timestep, cfm_method = self.cfm_method)
        elif self.cfm_method == 'independent_cfm_mix':
            HR_audio = self.sample(cond = cond, time_steps = timestep, cfm_method = self.cfm_method)

        HR_audio = HR_audio.squeeze(1) # [1, T]

        # post-proceesing w.r.s.t audio-level
        HR_audio_pp = self.postproc.post_processing(HR_audio, cond, cond.size(-1)) # [1, T]
        return HR_audio_pp

    def set_cfm_method(self, cfm_method):
        self.cfm_method = cfm_method
        # torchdiffeq_ode_method
        # sigma

    @classmethod
    def from_local(cls, ckpt_dir: Path, device) -> 'FlowHighSR':
        ckpt_dir = Path(ckpt_dir)
        voc = MelVoco(
            vocoder_config=ckpt_dir / "bigvgan.json",
            vocoder_path=ckpt_dir / "bigvgan.pt",
        )

        SR_generator = FLowHigh(
            dim_in = voc.n_mels,
            audio_enc_dec = voc,
            depth =2, # args.n_layers,
        )
        SR_generator = SR_generator.cuda().eval()

        cfm_wrapper=cls(
            flowhigh=SR_generator,
            # cfm_method = args.cfm_method,
            # torchdiffeq_ode_method=args.ode_method,
            # sigma = args.sigma,
        )
        # checkpoint load
        model_checkpoint = torch.load(
            ckpt_dir / "flowhigh.pt",
            map_location=device
        )
        cfm_wrapper.load_state_dict(model_checkpoint['model']) # dict_keys(['model', 'optim', 'scheduler'])
        cfm_wrapper = cfm_wrapper.cuda().eval()
        return cfm_wrapper
