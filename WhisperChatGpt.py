import io
from pydub import AudioSegment
import speech_recognition as sr
import whisper
import tempfile
import os
import easygui as eg
import torch
import numpy as np
from gtts import gTTS
import speechbrain as sb
from speechbrain.dataio.dataio import read_audio
import librosa
import librosa.display
import matplotlib.pyplot as plt
import playsound
from revChatGPT.revChatGPT import Chatbot
import traceback


def get_audio(model='base', english=True, verbose=False, energy=300, pause=1, dynamic_energy=False, save_file=False):
    if save_file:
        temp_dir = tempfile.mkdtemp()
        save_path = os.path.join(temp_dir, "temp.wav")
    else:
        save_path = None
    # there are no english models for large
    if model != "large" and english:
        model = model + ".en"
    audio_model = whisper.load_model(model)

    # load the speech recognizer and set the initial energy threshold and pause threshold
    r = sr.Recognizer()
    r.energy_threshold = energy
    r.pause_threshold = pause
    r.dynamic_energy_threshold = dynamic_energy

    with sr.Microphone(sample_rate=16000) as source:
        print("Say something!")
        #while True:
        # get and save audio to wav file
        audio = r.listen(source)
        if save_file:
            data = io.BytesIO(audio.get_wav_data())
            audio_clip = AudioSegment.from_file(data)
            audio_clip.export(save_path, format="wav")
            audio_data = save_path
        else:
            torch_audio = torch.from_numpy(
                np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
            audio_data = torch_audio

        if english:
            result = audio_model.transcribe(audio_data, language='english')
        else:
            result = audio_model.transcribe(audio_data)

        if not verbose:
            predicted_text = result["text"]
            print("You said: " + predicted_text)
            said = predicted_text
        else:
            print(result)
            said = result['text']
        return said

#main()

def speak(text,filename='_'):
    tts = gTTS(text=text, lang='en')
    filename = f'voice_{filename}.mp3'
    tts.save(filename)
    playsound.playsound(filename)





speak("Hello, how are you?")
said = get_audio(model='base', english=True, save_file=False)
#print(said)

if 'how are you' in said:
    speak(f'you said:{said}', filename='said')
    speak("I am fine, thank you. how can I help you?")
else:

    try:
        chatbot = Chatbot({

        })
        response = chatbot.get_chat_response(said)
        print(response)
        response_message = response['message']
        print(response_message)
        speak(response_message,filename='response')

    except:
        traceback.print_exc()
        print('error')
        speak('error',filename='error')

