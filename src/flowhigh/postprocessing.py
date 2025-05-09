import torch
from torchaudio.transforms import Spectrogram, InverseSpectrogram


class PostProcessing:
    def __init__(self, rank):
        self.stft = Spectrogram(2048, hop_length=480, win_length=2048, power=None, pad_mode='constant').cuda(rank)
        self.istft = InverseSpectrogram(2048, hop_length=480, win_length=2048, pad_mode='constant').cuda(rank)

    def get_cutoff_index(self, spec, threshold=0.99):
        energy = torch.cumsum(torch.sum(spec.squeeze().abs(), dim=-1), dim=0)
        threshold = energy[-1] * threshold
        for i in range(1, energy.size(0)):
            if energy[-i] < threshold:
                return energy.size(0) - i
        return 0

    def post_processing(self, pred, src, length):
        # pred, src : [1, Time]
        assert len(pred.shape) == 2 and len(src.shape) == 2

        spec_pred = self.stft(pred) # [1, Channel, Time]
        spec_src  = self.stft(src) # [1, Channel, Time]

        # energy cutoff of spec_src
        cr = self.get_cutoff_index(spec_src)

        # Replacement
        spec_result = torch.empty_like(spec_pred)
        min_time_dim = min(spec_pred.size(-1), spec_src.size(-1))

        spec_result = spec_result[:, :, :min_time_dim]
        spec_pred = spec_pred[:, :, :min_time_dim]
        spec_src = spec_src[:, :, :min_time_dim]

        spec_result[:,cr:, ...] = spec_pred[:, cr:, ...]
        spec_result[:, :cr, ...] = spec_src[:, :cr, ...]

        audio = self.istft(spec_result, length=length)
        audio = audio / torch.abs(audio).max() * 0.99
        return audio

    def post_processing_with_phase(self, pred, src, length):
        # pred, src : [1, Time]
        assert len(pred.shape) == 2 and len(src.shape) == 2

        spec_pred = self.stft(pred) # [1, Channel, Time]
        spec_src  = self.stft(src) # [1, Channel, Time]

        batch = spec_pred.shape[0]
        cr = self.get_cutoff_index(spec_src)

        # Replacement
        spec_result = torch.empty_like(spec_pred)
        min_time_dim = min(spec_pred.size(-1), spec_src.size(-1))

        spec_result = spec_result[:, :, :min_time_dim]
        spec_pred = spec_pred[:, :, :min_time_dim]
        spec_src = spec_src[:, :, :min_time_dim]

        pred_mag = torch.abs(spec_pred[:, cr:, ...])
        src_phase = torch.angle(spec_src[:, :cr, ...])

        # Replicate phase information to match the dimensions of spec_pred
        num_repeats = (spec_pred.size(1) - cr) // cr + 1
        replicate_phase = src_phase.repeat(batch, num_repeats, 1)
        replicate_phase = replicate_phase[:, - (spec_pred.size(1) - cr):, ...]
        print(pred_mag.size())
        print(replicate_phase.size())

        x = torch.cos(replicate_phase)
        y = torch.sin(replicate_phase)

        spec_result[:, cr:, ...] = pred_mag * (x + 1j * y)
        spec_result[:, :cr, ...] = spec_src[:, :cr, ...]

        audio = self.istft(spec_result, length=length)
        audio = audio / torch.abs(audio).max() * 0.99
        return audio, src_phase ,replicate_phase


    # For mel repalcement
    def _locate_cutoff_freq(self, stft, percentile=0.985):
        def _find_cutoff(x, percentile=0.95):
            percentile = x[-1] * percentile
            for i in range(1, x.shape[0]):
                if x[-i] < percentile:
                    return x.shape[0] - i
            return 0

        magnitude = torch.abs(stft)
        energy = torch.cumsum(torch.sum(magnitude, dim=0), dim=0)
        return _find_cutoff(energy, percentile)

    def mel_replace_ops(self, samples, input):
        for i in range(samples.size(0)):
            cutoff_melbin = self._locate_cutoff_freq(torch.exp(input[i]))
            samples[i][..., :cutoff_melbin] = input[i][..., :cutoff_melbin]
        return samples
