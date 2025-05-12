import torchaudio as ta
from flowhigh import FlowHighSR

TARGET_SR = 48000
INPUT_FILE = "infer-00.wav"
OUTPUT_FILE = "syn_out/infer-00-hr.wav"

_wav, _sr = ta.load(INPUT_FILE)
model = FlowHighSR.from_pretrained(device="cuda")
wav_hr = model.generate(_wav, _sr, TARGET_SR)
ta.save(OUTPUT_FILE, wav_hr.cpu(), TARGET_SR)
