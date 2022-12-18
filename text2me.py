import io
from pydub import AudioSegment
import speech_recognition as sr
import whisper
import tempfile
import os
import click
import torch
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import speechbrain as sb
from speechbrain.dataio.dataio import read_audio
from gtts import gTTS
import playsound
import time
from datetime import datetime
import geopy
import json

import openai
import chronological as ch
from chronological import read_prompt, cleaned_completion, main
from PIL import Image
import requests
import traceback

# find the current external ip address of this machine
def get_ip():
    #import requests
    try:
        ip = requests.get('https://api.ipify.org').text
        print('My public IP address is: {}'.format(ip))
        return ip

    except:
        traceback.print_exc()
    print('My public IP address is: {}'.format(ip))
    return ip


# get location based on ip address
def location(ip):
    import requests
    url = 'http://ip-api.com/json/'
    response = requests.get(url+ip)
    data = response.json()
    print(data)
    return data

# write a function to get the current location
def get_current_location():
    from geopy.geocoders import Nominatim
    geolocator = Nominatim(user_agent="Whisper2me")
    myip = get_ip()
    loc = geolocator.geocode(location(myip)['city'])
    return loc.address


# write a function to get the weather for a place

def get_weatherFor(place):
    import requests
    import json
    api_key = json.load(open('keys.json'))['openweather']
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "appid=" + api_key + "&q=" + place
    response = requests.get(complete_url)
    x = response.json()
    if x["cod"] != "404":
        y = x["main"]
        current_temperature = y["temp"]
        current_pressure = y["pressure"]
        current_humidiy = y["humidity"]
        z = x["weather"]
        weather_description = z[0]["description"]
        return f'The temperature is {current_temperature} Kelvin, the pressure is {current_pressure} hPa, the humidity is {current_humidiy} %, and the weather is {weather_description}'
    else:
        return " City Not Found "



# write a function to get the weather for the day
def get_current_weather():
    import requests
    import json
    api_key = json.load(open('keys.json'))['openweather']
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    myip = get_ip()
    city_name = location(myip)['city']
    complete_url = base_url + "appid=" + api_key + "&q=" + city_name + "&units=imperial"
    response = requests.get(complete_url)
    x = response.json()
    if x["cod"] != "404":
        y = x["main"]
        current_temperature = y["temp"]
        current_pressure = y["pressure"]
        current_humidiy = y["humidity"]
        z = x["weather"]
        weather_description = z[0]["description"]
        return f'The temperature is {current_temperature} Fahrenheit, the pressure is {current_pressure} hPa, the humidity is {current_humidiy} %, and the weather is {weather_description}'
    else:
        return " City Not Found "

def speak(text, filename='_'):
    try:
        if text is not None:
            timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            tts = gTTS(text=text, lang='en', tld='co.uk')
            filename = f'voice_mimic_{filename}-{timestamp}.mp3'
            folder = os.path.join('logs','voice')
            if not os.path.exists(folder):
                os.makedirs(folder)
            filepath = os.path.join(folder, filename)
            tts.save(filepath)
            playsound.playsound(filepath,block=False)
        else:

            print('No text to speak')
    except Exception as e:
        print(e)
        print(traceback.format_exc())


def get_text_completion(prompt):
    openai.api_key = json.load(open('keys.json'))['openai']
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens=500,
        top_p=1,
        frequency_penalty=1,
        presence_penalty=0.6,
        #stop=["\n", " Human:", " AI:"]
    )
    return response.choices[0].text


def get_code_completion(prompt):
    openai.api_key = json.load(open('keys.json'))['openai']
    response = openai.Completion.create(
        engine="code-davinci-002",
        prompt=prompt,
        temperature=0.5,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0.5,
        presence_penalty=0.6,
        #stop=["\n", " Human:", " AI:"]
    )
    return response.choices[0].text


def generate_Dalle_images(text, n=4):
    text = text.lower().replace('images', '').replace('image', '').strip()
    openai.api_key = json.load(open('keys.json'))['openai']
    response = openai.Image.create(prompt=text, n=n, size="512x512")
    name = text[:20].replace(' ', '_')
    for i in range(n):
        print(response['data'][i].url)

        img = Image.open(requests.get(response['data'][i].url, stream=True).raw)
        filename = f'{name}_{i}.png'
        folder = os.path.join('logs','images','dalle')
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder, filename)
        img.save(filepath)
        try:
            show =Image.open(filepath)
            show.show()
        except:
            pass

# write a function to generate images from stable diffusion using hugging face pipeline for stable diffusion
def generate_SD_images(text, n=4):
    from diffusers import StableDiffusionPipeline
    import torch

    text = text
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe.to('cuda')
    pipe.enable_xformers_memory_efficient_attention()
    pipe.safety_checker = None
    # params:
    height = 512
    width = 512
    num_inference_steps = 120
    guidance_scale = 0.75
    negative_prompt = None
    num_images_per_prompt = 2
    eta = 0.0
    generator = None
    latents = None
    output_type = "pil"
    return_dict = True
    callback = None
    callback_steps = 0
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    for i in range(n):

        image = pipe(prompt=text,
                     #height = height,
                     #width=width,
                     num_inference_steps=num_inference_steps,
                     guidance_scale=guidance_scale,
                     #negative_prompt=negative_prompt,
                     #num_images_per_prompt=num_images_per_prompt,
                     #eta=eta,
                     #generator=generator,
                     #latents=latents,
                     #output_type=output_type,
                     #return_dict=return_dict,
                     #callback=callback,
                     #callback_steps=callback_steps,
                     ).images[0]

        name = text[:20].replace(' ', '_').replace(',', '').replace('.', '').replace('?', '').replace('!', '').replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace('{', '').replace('}', '').replace(':', '').replace(';', '').replace('\'', '').replace('\"', '').replace('`', '').replace('~', '').replace('=', '').replace('-', '').replace('_', '').replace('+', '').replace('*', '').replace('/', '').replace('\\', '').replace('|', '').replace('<', '').replace('>', '')
        filename = f'{name}-{timestamp}_{i}.png'
        folder = os.path.join('logs','images','stable_diffusion')
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder, filename)
        image.save(filepath)




def respond_with_google(text):
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
    model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')
    input_text = text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(input_ids, max_length=100, num_beams=5, early_stopping=True)
    print(tokenizer.decode(outputs[0]))
    return tokenizer.decode(outputs[0])

def play_a_song(prompt):
    import torch
    from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
    import torch
    from PIL import Image, ImageDraw
    import os
    import numpy as np
    from scipy.io.wavfile import read

    from riffusion_test.riffusion.riffusion_pipeline import RiffusionPipeline
    from riffusion_test.riffusion.datatypes import PromptInput, InferenceInput
    from riffusion_test.riffusion.audio import wav_bytes_from_spectrogram_image
    from PIL import Image
    import struct
    import random
    repo_id = 'riffusion/riffusion-model-v1'
    from diffusers import StableDiffusionPipeline
    from diffusers import StableDiffusionImg2ImgPipeline
    from audio import wav_bytes_from_spectrogram_image
    import playsound
    pipe = StableDiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16).to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    pipe2 = StableDiffusionImg2ImgPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
    pipe2 = pipe2.to("cuda")
    pipe2.enable_xformers_memory_efficient_attention()
    with torch.autocast("cuda"):
        image = pipe(prompt, width=768).images[0];
    wav = wav_bytes_from_spectrogram_image(image)
    filename = prompt.replace(" ", "_")[:20] + ".wav"
    filedir = os.path.join('logs','music')
    filepath = os.path.join(filedir, filename)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    with open(filepath, "wb") as f:
        f.write(wav[0].getbuffer())
    playsound.playsound(filepath)
    return filepath

def generate_a_song(prompt_1,prompt_2='',steps=25, num_iterations=2, feel='og_beat',seed=0):
    prompt = prompt_1
    prompt2 = prompt_2
    steps = steps
    num_iterations = num_iterations
    feel = feel
    seed = seed

    from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline, \
        StableDiffusionInpaintPipeline
    import torch
    from PIL import Image, ImageDraw
    import os
    import numpy as np
    from scipy.io.wavfile import read
    from riffusion.riffusion.riffusion_pipeline import RiffusionPipeline
    from riffusion.riffusion.datatypes import PromptInput, InferenceInput
    from riffusion.riffusion.audio import wav_bytes_from_spectrogram_image
    from PIL import Image
    import struct
    import random

    repo_id = "riffusion/riffusion-model-v1"

    model = RiffusionPipeline.from_pretrained(
        repo_id,
        revision="main",
        torch_dtype=torch.float16,
        safety_checker=lambda images, **kwargs: (images, False),
    )

    if torch.cuda.is_available():
        model.to("cuda")
    model.enable_xformers_memory_efficient_attention()

    pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(repo_id, torch_dtype=torch.float16,
                                                                  safety_checker=lambda images, **kwargs: (
                                                                  images, False), )
    pipe_inpaint.scheduler = DPMSolverMultistepScheduler.from_config(pipe_inpaint.scheduler.config)
    if torch.cuda.is_available():
        pipe_inpaint = pipe_inpaint.to("cuda")
    pipe_inpaint.enable_xformers_memory_efficient_attention()

    def get_init_image(image, overlap, feel):

        width, height = image.size
        init_image = Image.open(f"riffusion/seed_images/{feel}.png").convert("RGB")
        # Crop the right side of the original image with `overlap_width`
        cropped_img = image.crop((width - int(width * overlap), 0, width, height))
        init_image.paste(cropped_img, (0, 0))

        return init_image

    def get_mask(image, overlap):

        width, height = image.size

        mask = Image.new("RGB", (width, height), color="white")
        draw = ImageDraw.Draw(mask)
        draw.rectangle((0, 0, int(overlap * width), height), fill="black")
        return mask

    def i2i(prompt, steps, feel, seed):
        #   return pipe_i2i(
        #       prompt,
        #       num_inference_steps=steps,
        #       image=Image.open(f"riffusion/seed_images/{feel}.png").convert("RGB"),
        #       ).images[0]

        prompt_input_start = PromptInput(prompt=prompt, seed=seed)
        prompt_input_end = PromptInput(prompt=prompt, seed=seed)

        return model.riffuse(
            inputs=InferenceInput(
                start=prompt_input_start,
                end=prompt_input_end,
                alpha=1.0,
                num_inference_steps=steps),
            init_image=Image.open(f"riffusion/seed_images/{feel}.png").convert("RGB")
        )

    def outpaint(prompt, init_image, mask, steps):
        return pipe_inpaint(
            prompt,
            num_inference_steps=steps,
            image=init_image,
            mask_image=mask,
        ).images[0]

    def generate(prompt, steps, num_iterations, feel, seed):

        if seed == 0:
            seed = random.randint(0, 4294967295)

        num_images = num_iterations
        overlap = 0.5
        image_width, image_height = 512, 512  # dimensions of each output image
        total_width = num_images * image_width - (num_images - 1) * int(
            overlap * image_width)  # total width of the stitched image

        # Create a blank image with the desired dimensions
        stitched_image = Image.new("RGB", (total_width, image_height), color="white")


        # Initialize the x position for pasting the next image
        x_pos = 0

        image = i2i(prompt, steps, feel, seed)

        for i in range(num_images):
            # Generate the prompt, initial image, and mask for this iteration
            init_image = get_init_image(image, overlap, feel)
            mask = get_mask(init_image, overlap)

            # Run the outpaint function to generate the output image
            steps = 25
            image = outpaint(prompt, init_image, mask, steps)

            # Paste the output image onto the stitched image
            stitched_image.paste(image, (x_pos, 0))

            # Update the x position for the next iteration
            x_pos += int((1 - overlap) * image_width)

        wav_bytes, duration_s = wav_bytes_from_spectrogram_image(stitched_image)

        # mask = Image.new("RGB", (512, 512), color="white")
        # bg_image = outpaint(prompt, init_image, mask, steps)
        # bg_image.save("bg_image.png")
        timestamp = str(int(time.time()))
        filename = "bg_image"+prompt.replace(" ", "_")[:20] + timestamp + ".png"
        st_filename = "stitched_image"+prompt.replace(" ", "_")[:20] + timestamp + ".png"
        filedir = os.path.join('logs','music','bg_image')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        filepath = os.path.join(filedir, filename)
        st_filepath = os.path.join(filedir, st_filename)
        stitched_image.save(st_filepath)
        init_image.save(filepath)


        # return read(wav_bytes)

        filename = "music"+prompt.replace(" ", "_")[:20] + timestamp + ".wav"
        filedir = os.path.join('logs','music')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        filepath = os.path.join(filedir, filename)
        with open(filepath, "wb") as f:
            f.write(wav_bytes.read())

        playsound.playsound(filepath)

    def riffuse(steps, feel, init_image, prompt_start, seed_start, denoising_start=0.75, guidance_start=7.0,
                prompt_end=None, seed_end=None, denoising_end=0.75, guidance_end=7.0, alpha=0.5):

        prompt_input_start = PromptInput(prompt=prompt_start, seed=seed_start, denoising=denoising_start,
                                         guidance=guidance_start)

        prompt_input_end = PromptInput(prompt=prompt_end, seed=seed_end, denoising=denoising_end, guidance=guidance_end)

        input = InferenceInput(
            start=prompt_input_start,
            end=prompt_input_end,
            alpha=alpha,
            num_inference_steps=steps,
            seed_image_id=feel,

        )

        image = model.riffuse(inputs=input, init_image=init_image)

        wav_bytes, duration_s = wav_bytes_from_spectrogram_image(image)

        return wav_bytes, image

    def wav_list_to_wav(wav_list):

        # remove headers from the WAV files
        data = [wav.read()[44:] for wav in wav_list]

        # concatenate the data
        concatenated_data = b"".join(data)

        # create a new RIFF header
        channels = 1
        sample_rate = 44100
        bytes_per_second = channels * sample_rate
        new_header = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", len(concatenated_data) + 44 - 8, b"WAVE", b"fmt ", 16,
                                 1, channels, sample_rate, bytes_per_second, 2, 16, b"data", len(concatenated_data))

        # combine the header and data to create the final WAV file
        final_wav = new_header + concatenated_data
        return final_wav


    def generate_riffuse(prompt_start, steps, num_iterations, feel, prompt_end=None, seed_start=None, seed_end=None,
                         denoising_start=0.75, denoising_end=0.75, guidance_start=7.0, guidance_end=7.0):
        """Generate a WAV file of length seconds using the Riffusion model.
        Args:
            length (int): Length of the WAV file in seconds, must be divisible by 5.
            prompt_start (str): Prompt to start with.
            prompt_end (str, optional): Prompt to end with. Defaults to prompt_start.
            overlap (float, optional): Overlap between audio clips as a fraction of the image size. Defaults to 0.2.
            """

        # open the initial image and convert it to RGB
        try:
            init_image = Image.open(f"riffusion\\seed_images\\{feel}.png").convert("RGB")
        except:
            init_image = Image.open(f"riffusion\\seed_images\\vibes.png").convert("RGB")

        if prompt_end is None:
            prompt_end = prompt_start
        if seed_start == 0:
            seed_start = random.randint(0, 4294967295)
        if seed_end is None:
            seed_end = seed_start

        # one riffuse() generates 5 seconds of audio
        wav_list = []

        for i in range(int(num_iterations)):
            alpha = i / (num_iterations - 1)
            print(alpha)
            wav_bytes, image = riffuse(steps, feel, init_image, prompt_start, seed_start, denoising_start,
                                       guidance_start, prompt_end, seed_end, denoising_end, guidance_end, alpha=alpha)
            wav_list.append(wav_bytes)

            init_image = image

            seed_start = seed_end
            seed_end = seed_start + 1


        filename = "bg_image"+prompt_start.replace(" ", "_")[:20] + str(int(time.time())) + ".png"
        filedir = os.path.join('logs','music','bg_image')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        filepath = os.path.join(filedir, filename)
        init_image.save(filepath)

        filename = "music_"+prompt_start.replace(" ", "_")[:20] + str(int(time.time())) + ".wav"
        filename_txt = "music_"+prompt_start.replace(" ", "_")[:20] + str(int(time.time())) + ".txt"
        filedir = os.path.join('logs','music')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        filepath = os.path.join(filedir, filename)
        filepath_txt = os.path.join(filedir, filename_txt)
        with open(filepath, "wb") as f:
            f.write(wav_list_to_wav(wav_list))
        with open(filepath_txt, "w") as f:
            f.write(f'{prompt_start},{prompt_end},{steps},{num_iterations},{feel},{seed}')
        playsound.playsound(filepath, block=False)

    if prompt =="" or prompt is None:
        print("No prompt given")
        return None
    if prompt2 =="" or prompt2 is None:
        generate(prompt, steps, num_iterations, feel,seed)
    else:
        generate_riffuse(prompt, steps, num_iterations, feel, prompt_end=prompt2, seed_start=seed)




def split_text(text, n=4000):
    return [text[i:i+n] for i in range(0, len(text), n)]

def parse_said(text):
    text = text.lower()
    text =str.strip(text)

    if 'weather' in text:
        return get_current_weather()
    elif 'song' in text:
        text = text.lower()
        text = text.replace('play a song', '').replace('make a song', '')
        return play_a_song(text)
    elif 'image' in text:
        return generate_Dalle_images(text)
    elif 'code' in text:
        return get_code_completion(text)
    elif 'text' in text:
        return get_text_completion(text)
    elif 'google' in text:
        return respond_with_google(text)
    else:
        return 'I do not know how to do that yet'



def main():
    openaikey = json.load(open('keys.json'))['openai']
    openai.api_key = openaikey
    print(openai.Model.list())

    prompt = input('What do you want to do? ')
    filename_it = 0
    while prompt != 'exit' or prompt != 'stop':
        said = prompt.lower()
        if 'get time' in said:
            print('time triggered')
            filename_it += 1
            speak(f'The time is, {time.ctime()}')
            # convert time to epoch time
            epoch_time = int(time.time())
            print(epoch_time)
            said = 'none'

        if 'date' in said:
            print('date triggered')
            filename_it += 1
            speak(f'Todays date is, {datetime.now().strftime("%d-%m-%Y")}')
            said = 'None'
        if 'weather' in said:
            print('weather triggered')
            filename_it += 1
            current_weather = get_current_weather()
            speak(f' {current_weather}')
            said = 'none'
        if 'generate images of' in said:
            print('image generation triggered')
            speak(f'Generating 4 images of {said[18:]}')
            generate_Dalle_images(said[18:])
            speak(f'Youre image of  {said[18:]}, have been generated')
            said = 'None'
        if 'generate' and 'images' in said:
            print('image generation with number triggered')
            #find the number which should be between generate and images, split between "generate" and "images"
            num = said.rsplit("generate").lsplit("images")[0]
            if num is not int:
                num = 4
            pre = len(said.split("images")[0])
            speak(f'Generating {num} images of {said[pre:]}')
            generate_Dalle_images(said[pre:], num)
            speak(f'Youre {said[pre:]}, have been generated')
            said = 'None'
        if 'stable diffuse' in said:
            print('stable diffusion triggered')
            speak(f'Generating 4 images of {said[15:]}')
            generate_SD_images(said[15:])
            speak(f'Youre images of  {said[24:]}, have been generated')
            said = 'None'
        if 'help' and 'code' in said:
            print('help code triggered')
            pre = len(said.split("code")[0])
            speak(f'Sure, I can help you with generating code for {said[pre:]}')
            result = get_code_completion(said[pre:])
            speak(f'Here is some code for {said[pre:]}')
            print(result)
            said = 'None'
        if 'what' and 'help' in said:
            print('help triggered')
            speak(f'You can ask me to generate images, generate code, tell you the time, date, weather, or stop recording')
            said = 'None'
        if 'gpt' in said:
            print('gpt triggered')
            text = said
            response = get_text_completion(text)
            print(response)
            speak(response)
            said = 'None'
        if 'google' in said:
            print('hey g triggered')
            said = said.replace('google', '')
            response = respond_with_google(said)
            print(response)
            said = 'None'
        if 'make a clip' in said or 'generate a clip'in said:
            print('clip triggered')
            said = said.replace('make a clip ', '')
            said = said.replace('play a clip ', '')
            said = said.replace('with ', '')
            said = str.strip(said)
            response = play_a_song(said)
            print(response)
            said = 'None'
        if 'make a song' in said or 'play a song'in said:
            print('song triggered')
            prompt = input('prompt_1,prompt_2="",steps=25, num_iterations=2, feel="og_beat",seed=0')
            args = prompt.split(',')
            prompt_1 = args[0]
            prompt_2 = args[1]
            steps = int(args[2])
            num_iterations = int(args[3])
            feel = args[4]
            seed = int(args[5])
            response = generate_a_song(prompt_1, prompt_2, steps, num_iterations, feel, seed)
            print(response)
            said = 'None'
        if said.lower() == 'none':
            prompt = input('What do you want to do? ')
        else:
            try:
                print('gpt3 triggered as after thought')
                text = said
                more = input('Do you want to talk?')
                if more == 'yes':
                    response = get_text_completion(text)
                    speak(response)
                    said = 'none'
                elif more == 'no':
                    response = get_text_completion(text)
                    print(response)
                    said = 'none'
                    prompt = input('What do you want to do? ')
                else:
                    speak('Ok')
                    more = input('Do you want to add more text with google?')
                    if more == 'yes':
                        response = respond_with_google(text)
                        print(response)
                        said = 'none'
                        prompt = input('What do you want to do? ')
                    else:
                        print('Ok')
                        said = 'none'
                        prompt = input('What do you want to do? ')
            except:
                print('gpt failed')
                print('local google model triggered')
                text = said
                response = respond_with_google(text)
                print(response)
                said = 'none'
                #speak(response)


    else:
        print(result)


main()