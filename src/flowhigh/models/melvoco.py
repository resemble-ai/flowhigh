from librosa.filters import mel as librosa_mel_fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from einops import rearrange

from .bigvgan.init_vocoder import init_bigvgan
from .modules import spectral_normalize_torch


mel_basis = {}
hann_window = {}


class MelVoco(nn.Module):
    def __init__(
        self,
        *,
        log=True,
        n_mels=256,
        sampling_rate=48000,
        f_max=24000,
        f_min=20,
        n_fft=2048,
        win_length=2048,
        hop_length=480,
        vocoder="bigvgan",
        vocoder_config='./vocoder_config.json',
        vocoder_path=None
    ):
        super().__init__()
        self.log = log
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.f_max = f_max
        self.f_min = f_min
        self.win_length = win_length
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate

        if vocoder == 'bigvgan':
            self.vocoder_name = vocoder
            self.vocoder = init_bigvgan(vocoder_config, vocoder_path, vocoder_freeze=True)
        else:
            raise ValueError("unsuitable vocoder name")

    @property
    def downsample_factor(self):
        raise NotImplementedError

    @property
    def latent_dim(self):
        return self.n_mels

    def encode(self, audio):
        if torch.min(audio) < -1.:
            print('min value is ', torch.min(audio))
        if torch.max(audio) > 1.:
            print('max value is ', torch.max(audio))

        global mel_basis, hann_window
        if self.f_max not in mel_basis:
            mel = librosa_mel_fn(
                sr=self.sampling_rate,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                fmin=self.f_min,
                fmax=self.f_max,
            )
            mel_basis[str(self.f_max)+'_'+str(audio.device)] = torch.from_numpy(mel).float().to(audio.device)
            hann_window[str(audio.device)] = torch.hann_window(self.win_length).to(audio.device)

        audio = torch.nn.functional.pad(audio.unsqueeze(1), (int((self.n_fft-self.hop_length)/2), int((self.n_fft-self.hop_length)/2)), mode='reflect')
        audio = audio.squeeze(1)

        # complex tensor as default, then use view_as_real for future pytorch compatibility
        spec = torch.stft(audio, self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=hann_window[str(audio.device)],
                        center=False, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

        spec = torch.matmul(mel_basis[str(self.f_max)+'_'+str(audio.device)], spec)
        spec = spectral_normalize_torch(spec)
        spec = rearrange(spec, 'b d n -> b n d')
        return spec

    def encode_torchaudio(self, audio):

        stft_transform = T.Spectrogram(
            n_fft = self.n_fft,
            win_length = self.win_length,
            hop_length = self.hop_length,
            window_fn = torch.hann_window
        ).cuda()

        audio = audio.cuda()
        spectrogram = stft_transform(audio)

        mel_transform = T.MelScale(
            n_mels = self.n_mels,
            sample_rate = self.sampling_rate,
            n_stft = self.n_fft // 2 + 1,
            f_max = self.f_max
        ).cuda()

        spec = mel_transform(spectrogram)

        if self.log:
            spec = T.AmplitudeToDB()(spec)
        spec = rearrange(spec, 'b d n -> b n d')
        return spec

    def decode(self, mel):
        mel = rearrange(mel, 'b n d -> b d n')

        # if self.log:
        #     mel = DB_to_amplitude(mel, ref = 1., power = 0.5)

        if self.vocoder_name == 'bigvgan':
            return self.vocoder.forward(mel)

