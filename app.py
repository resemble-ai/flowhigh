# import numba, numpy, scipy, librosa, matplotlib, torch, torchaudio
from flowhigh import FlowHighSR
import gradio as gr

TARGET_SR = 48000
INPUT_FILE = "infer-00.wav"
OUTPUT_FILE = "syn_out/infer-00-hr.wav"

model = FlowHighSR.from_local("checkpoints", device="cuda")


def generate(audio, sr_out):
    sr_in, audio = audio
    print(sr_in)
    print(audio)
    wav = model.generate(
        audio, sr_in, sr_out
    )
    return TARGET_SR, wav.detach().cpu().squeeze(0).numpy()


demo = gr.Interface(
    generate,
    [
        gr.Audio(sources="upload", type="numpy", label="Input audio file"),
        gr.Radio([16000, 22050, 24000, 32000, 44100, 48000], value=48000),
    ],
    "audio",
)


if __name__ == "__main__":
    demo.launch()
