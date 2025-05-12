import torchaudio as ta
from flowhigh import FlowHighSR

TARGET_SR = 48000
INPUT_FILE = "LOW-RES-AUDIO.wav"
OUTPUT_FILE = "OUTPUT.wav"

model = FlowHighSR.from_pretrained(device="cuda")

wav, sr_in = ta.load(INPUT_FILE)
wav_hr = model.generate(wav, sr_in, TARGET_SR)
ta.save(OUTPUT_FILE, wav_hr.cpu(), TARGET_SR)
