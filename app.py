from flowhigh import FlowHighSR
import gradio as gr


model = FlowHighSR.from_local("checkpoints", device="cuda")


def generate(audio, sr_out):
    sr_in, audio = audio
    print(sr_in)
    print(audio)
    wav = model.generate(
        audio, sr_in, sr_out
    )
    return sr_out, wav.detach().cpu().squeeze(0).numpy()


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
