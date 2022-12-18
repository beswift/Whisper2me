import os
import torch
from diffusers import StableDiffusionPipeline
from audio import wav_bytes_from_spectrogram_image
import easygui as eg

pipe = StableDiffusionPipeline.from_pretrained("riffusion/riffusion-model-v1", torch_dtype=torch.float16).to("cuda")
pipe.enable_xformers_memory_efficient_attention()

promptd = "a slow and technical guitar jam with indian tones"  # @param {type: 'string'}

with torch.autocast("cuda"):
    image = pipe(promptd, height=512, width=512).images[0];
wav = wav_bytes_from_spectrogram_image(image)
with open("output.wav", "wb") as f:
    f.write(wav[0].getbuffer())
#IPython.display.Audio("output.wav", rate=44100)


def play_a_song(prompt):
    import torch
    from diffusers import StableDiffusionPipeline
    from audio import wav_bytes_from_spectrogram_image
    import playsound
    pipe = StableDiffusionPipeline.from_pretrained("riffusion/riffusion-model-v1", torch_dtype=torch.float16).to("cuda")
    with torch.autocast("cuda"):
        image = pipe(prompt, height=512, width=512).images[0];
    wav = wav_bytes_from_spectrogram_image(image)
    filename = prompt.replace(" ", "_")[:] + ".wav"
    filedir = "os.path.join('logs','music')"
    filepath = os.path.join(filedir, filename)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    with open(filepath, "wb") as f:
        f.write(wav[0].getbuffer())
    playsound.playsound(filepath)
    return filepath



prompt = eg.enterbox("Enter a prompt for the song")
play_a_song(prompt)