from pathlib import Path
from functools import wraps
from einops import rearrange
from beartype import beartype
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torchaudio
import librosa
import scipy
from scipy.signal import butter, cheby1, cheby2, ellip, bessel, sosfiltfilt, resample_poly
import numpy as np
import random

# utilities

def exists(val):
    return val is not None

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# dataset functions

class AudioDataset(Dataset):
    @beartype
    def __init__(
        self,
        folder,
        audio_extension = ".wav",
        mode = None,
        downsampling = str(),
    ):
        super().__init__()
        path = Path(folder)
        assert path.exists(), 'folder does not exist'

        self.audio_extension = audio_extension
        self.downsampling = downsampling
        self.mode = mode
        files = list(path.glob(f'**/*{audio_extension}'))
        assert len(files) > 0, 'no files found'
        self.files = files


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]

        if self.downsampling == 'torchaudio':
            wave, sr = torchaudio.load(file) # [1, Time]
            wave = rearrange(wave, '1 ... -> ...') # [Time]
            length = wave.shape[-1]
            return wave, length

        elif self.downsampling == 'librosa':
            wave, sr = librosa.load(file, sr=None, mono=True) # [Time] 
            wave /= np.max(np.abs(wave))
            nyq = sr // 2
            min_value = 4000
            max_value = 32000
            step = 1000
            sampling_rates = list(range(min_value, max_value + step, step))
            random_sr = random.choice(sampling_rates)
            
            if self.mode == 'valid':
                order = 8
                ripple = 0.05
            else:
                order = random.randint(1, 11)
                ripple = random.choice([1e-9, 1e-6, 1e-3, 1, 5])

            highcut = random_sr // 2
            hi = highcut / nyq
            sos = cheby1(order, ripple, hi, btype='lowpass', output='sos')
            d_HR_wave = sosfiltfilt(sos, wave)
            down_cond = librosa.resample(d_HR_wave, sr, random_sr, res_type='soxr_hq')
            up_cond = librosa.resample(down_cond, random_sr, sr, res_type='soxr_hq')

            if len(up_cond) < len(wave):
                up_cond = np.pad(wave, (0, len(wave) - len(up_cond)), 'constant', constant_values=0)
            elif len(up_cond) > len(wave):
                up_cond = up_cond[:len(wave)]

            length = wave.shape[-1]

            if self.mode == 'valid':
                return torch.from_numpy(wave).float(), length
            return torch.from_numpy(wave).float(), length, torch.from_numpy(up_cond).float(), random_sr
        

        elif self.downsampling == 'scipy':

            wave, sr = librosa.load(file, sr=None, mono=True)
            wave /= np.max(np.abs(wave))
            nyq = sr // 2
            min_value = 4000
            max_value = 32000
            step = 1000
            sampling_rates = list(range(min_value, max_value + step, step))
            random_sr = random.choice(sampling_rates)
            
            if self.mode == 'valid':
                order = 8
                ripple = 0.05
            
            else:
                order = random.randint(1, 11)
                ripple = random.choice([1e-9, 1e-6, 1e-3, 1, 5])

            highcut = random_sr // 2
            hi = highcut / nyq

            sos = cheby1(order, ripple, hi, btype='lowpass', output='sos')
            d_HR_wave = sosfiltfilt(sos, wave)
            down_cond = resample_poly(d_HR_wave, random_sr, sr)
            up_cond = resample_poly(down_cond, sr, random_sr)

            if len(up_cond) < len(wave):
                up_cond = np.pad(wave, (0, len(wave) - len(up_cond)), 'constant', constant_values=0)
            elif len(up_cond) > len(wave):
                up_cond = up_cond[:len(wave)]
        
            length = wave.shape[-1]
            
            if self.mode == 'valid':
                return torch.from_numpy(wave).float(), length
            
            up_cond = torch.from_numpy(up_cond.copy()).float()
            wave = torch.from_numpy(wave.copy()).float()
            return wave,  length, up_cond, random_sr


# dataloader functions

def collate_one_or_multiple_tensors(fn):
    @wraps(fn)
    def inner(data):
        is_one_data = not isinstance(data[0], tuple)
        if is_one_data:
            data = fn(data)
            return (data,)
        outputs = []
        
        for index, datum in enumerate(zip(*data)):
            
            if index == 1: # length 
                output = torch.tensor(datum, dtype=torch.long)
            elif index == 2: # up_cond wav
                output = fn(datum)
            elif index == 3:
                output = list(datum)
            else:
                output = fn(datum)  
            outputs.append(output)
        return tuple(outputs)
    return inner

@collate_one_or_multiple_tensors
def curtail_to_shortest_collate(data):
    min_len = min(*[datum.shape[0] for datum in data])
    data = [datum[:min_len] for datum in data]
    return torch.stack(data)

@collate_one_or_multiple_tensors
def pad_to_longest_fn(data):
    return pad_sequence(data, batch_first = True)

def get_dataloader(ds, pad_to_longest = True, **kwargs):
    collate_fn = pad_to_longest_fn if pad_to_longest else curtail_to_shortest_collate
    return DataLoader(ds, collate_fn = collate_fn, num_workers = 8, persistent_workers=True, **kwargs)
