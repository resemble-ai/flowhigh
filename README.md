# Installation
`pip install git+https://github.com/resemble-ai/flowhigh.git@dev`


# Usage
```python
import torchaudio as ta
from flowhigh import FlowHighSR

TARGET_SR = 48000
INPUT_FILE = "infer-00.wav"
OUTPUT_FILE = "syn_out/infer-00-hr.wav"

_wav, _sr = ta.load(INPUT_FILE)
model = FlowHighSR.from_local("checkpoints", device="cuda")
wav_hr = model.generate(_wav, _sr, TARGET_SR)
ta.save(OUTPUT_FILE, wav_hr.cpu(), TARGET_SR)
```
See `example.py`.

<hr/>

# FLowHigh: Towards Efficient and High-Quality Audio Super-Resolution with Single-Step Flow Matching

## The official implementation of FLowHigh

<img src="./image1.jpg" align="center" width="300" height="300">

## <a src="https://img.shields.io/badge/eess.AS-2501.04926-b31b1b?logo=arxiv&logoColor=red" href="https://arxiv.org/abs/2501.04926"> <img src="https://img.shields.io/badge/eess.AS-2501.04926-b31b1b?logo=arxiv&logoColor=red"></a>  [![Static Badge](https://img.shields.io/badge/Demos-FLowHigh-brightred?style=flat)](https://jjunak-yun.github.io/FLowHigh)
**Jun-Hak Yun, Seung-Bin Kim, Seong-Whan Lee**

## Clone our repository
```
git clone https://github.com/jjunak-yun/FLowHigh_code.git
cd FLowHigh_code
```

## Install the requirements
```
pip install -r requirements.txt
```

## Data preparation
* Download the [VCTK](https://datashare.ed.ac.uk/handle/10283/3443) dataset.
* Remove speakers `p280` and `p315` from the dataset.
* Create a `train` directory and a `test` directory, then split the dataset accordingly.
* Update the `data_path` in the `configs/config.json` file with the path to your newly created `train` directory.

## Training
* To adjust the training conditions, modify the `configs/config.json` file according to your preferences.
```
CUDA_VISIBLE_DEVICES=0 python train.py
```

## Pre-trained checkpoints
[FLowHigh_indep_adaptive_400k](https://drive.google.com/drive/folders/1clsJ3bFTZSCLOb4I0wk0l6FL9Yh33Vof?usp=sharing) : This is the main model from our paper, an audio super-resolution model designed to reconstruct low-resolution audio into high-resolution audio at 48 kHz.<br><br>
[FLowHigh_basic_400k](https://drive.google.com/drive/folders/1IxOhrPiDt3eM6aLhiZ1x9Cu49TaRxu-W?usp=sharing) : This model adopts conditional probability path based on [basic flow-matching](https://arxiv.org/abs/2210.02747) and is also an audio super-resolution model that reconstructs low-resolution audio into high-resolution audio at 48 kHz.



## Inference of audio
* Prepare the checkpoint of the trained model.
* Prepare a downsampled audio sample with a sampling rate smaller than 48 kHz (e.g., 12 kHz, 16 kHz). <br>**Note**: If you wish to match the experimental setup in our paper, use `scipy.resample_poly()` to downsample the audio.
* Run the following command:
```
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --input_path {downsampled_audio_path} --output_path {save_output_audio_path} \
    --target_sampling_rate 48000 --up_sampling_method scipy --architecture='transformer' \
    --time_step 1 --ode_method={ode_solver} --cfm_method={cfm_path} --sigma 0.0001 \
    --model_path {model_checkpoint_path} \
    --n_layers 2 --n_heads 16 --dim_head 64 \
    --n_mels 256 --f_max 24000 --n_fft 2048 --win_length 2048 --hop_length 480 \
    --vocoder 'bigvgan' --vocoder_path='/PATH/vocoder/BIGVGAN/checkpoint/g_48_00850000' \
    --vocoder_config_path='/PATH/vocoder/BIGVGAN/config/bigvgan_48khz_256band_config.json' \
```

| Parameter Name       | Description                                                                                      |
|----------------------|--------------------------------------------------------------------------------------------------|
| --time_step               | The number of steps for solving the ODE (Ordinary Differential Equation). <br>In our paper, we utilized a single-step approach (`time_step=1`). <br>While increasing `time_step` generally enhances the quality, in our case, the improvement was not significantly noticeable.|
| --ode_method       | Choose between `euler` or `midpoint`. <br>The `midpoint` method improves performance but doubles the NFEs (Number of Function Evaluations). <br>**Recommendation**: Despite the increase in NFE, we recommend using the `midpoint` method for better performance. <br>**Note**: The choice of `ode_method` is independent of the trainind settings.|
| --cfm_method       | Sets the Conditional Probability Paths. <br>In our paper, we used the path `independent_cfm_adaptive`. <br>Other available options include `basic_cfm`(<https://arxiv.org/abs/2210.02747>) and `independent_cfm_constant`(<https://arxiv.org/abs/2302.00482>).|
| --sigma  | Influences the path setting. <br>Ensure you use the same value for `sigma` as was used during training. |

## To-do list
- [x] add base training code
- [x] add requirements.txt
- [x] upload pre-trained checkpoint for independent_cfm_adaptive
- [x] upload pre-trained checkpoint for basic_cfm
- [ ] optimize the training speed


## References
This implementation was developed based on the following repository:
* Voicebox(unofficial pytorch implementation): <https://github.com/lucidrains/voicebox-pytorch.git> (for architecture backbone)
* Fre-painter: <https://github.com/FrePainter/code.git> (for audio super-resolution implementation)
* TorchCFM: <https://github.com/atong01/conditional-flow-matching.git> (for CFM logic)
* BigVGAN: <https://github.com/NVIDIA/BigVGAN.git> (for pre-trained vocoder)
* Nu-wave2: <https://github.com/maum-ai/nuwave2.git> (for data processing)
