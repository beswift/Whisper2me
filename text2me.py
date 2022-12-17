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
    complete_url = base_url + "appid=" + api_key + "&q=" + city_name + "&units=fahrenheit"
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
            playsound.playsound(filepath)
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
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
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
        max_tokens=250,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        #stop=["\n", " Human:", " AI:"]
    )
    return response.choices[0].text


def generate_images(text, n=4):
    openai.api_key = json.load(open('keys.json'))['openai']
    response = openai.Image.create(prompt=text, n=n, size="512x512")
    name = text.split(' ')[0]
    for i in range(n):
        print(response['data'][i].url)

        img = Image.open(requests.get(response['data'][i].url, stream=True).raw)
        img.save(f'img_{name}-{i}.png')

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
    from diffusers import StableDiffusionPipeline
    from audio import wav_bytes_from_spectrogram_image
    import playsound
    pipe = StableDiffusionPipeline.from_pretrained("riffusion/riffusion-model-v1", torch_dtype=torch.float16).to("cuda")
    with torch.autocast("cuda"):
        image = pipe(prompt, height=512, width=512).images[0];
    wav = wav_bytes_from_spectrogram_image(image)
    filename = prompt.replace(" ", "_")[:8] + ".wav"
    filedir = os.path.join('logs','music')
    filepath = os.path.join(filedir, filename)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    with open(filepath, "wb") as f:
        f.write(wav[0].getbuffer())
    playsound.playsound(filepath)
    return filepath



def main():
    openaikey = json.load(open('keys.json'))['openai']
    openai.api_key = openaikey
    print(openai.Model.list())

    prompt = input('What do you want to do? ')
    filename_it = 0
    while prompt != 'exit' or prompt != 'stop':
        said = prompt.lower()
        if 'time' in said:
            print('time triggered')
            filename_it += 1
            speak(f'The time is, {time.ctime()}')
            said = 'None'
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
            said = 'None'
        if 'generate images of' in said:
            print('image generation triggered')
            speak(f'Generating 4 images of {said[24:]}')
            generate_images(said[24:])
            speak(f'Youre image of  {said[24:]}, have been generated')
            said = 'None'
        if 'generate' and 'images' in said:
            print('image generation with number triggered')
            #find the number which should be between generate and images, split between "generate" and "images"
            num = said.rsplit("generate")
            if num is not int:
                num = 4
            pre = len(said.split("images")[0])
            speak(f'Generating {num} images of {said[pre:]}')
            generate_images(said[pre:], num)
            speak(f'Youre {said[pre:]}, have been generated')
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
        song = ['make a song', 'play a song', 'make me a song', 'play me a song']
        if said in song:
            print('song triggered')
            said = said.replace('make a song ', '')
            said = said.replace('play a song ', '')
            said = said.replace('with ', '')
            said = str.strip(said)
            response = play_a_song(said)
            print(response)
            said = 'None'
        if said.lower() is 'none':
            said = input('What do you want to do? ')
        else:
            try:
                print('gpt triggered as after thought')
                text = said
                more = input('Do you want to talk?')
                if more == 'yes':
                    response = get_text_completion(text)
                    speak(response)
                    said = None
                elif more == 'no':
                    response = get_text_completion(text)
                    print(response)
                    said = None
                else:
                    speak('Ok')
                    more = input('Do you want to add more text with google?')
                    if more == 'yes':
                        response = respond_with_google(text)
                        print(response)
                        said = None
                    else:
                        print('Ok')
                        said = None
            except:
                print('gpt failed')
                print('local google model triggered')
                text = said
                response = respond_with_google(text)
                print(response)
                #speak(response)


    else:
        print(result)


main()