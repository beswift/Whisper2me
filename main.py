# Create a streamlit app to display the ui for recording audio from a users microphone or uploaded audio file and then transcribing the audio to text using the whisper model and displaying the results

#Then allow the user to transcribe the auido using another model supported by the huggingface transformers library or the speech_recognition library and display the results

#Then allow the user to save the results to a text file


# Import the necessary libraries

import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import os
import time
import speech_recognition as sr
#from transformers import pipeline
#from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import whisper
import tempfile
import io
from pydub import AudioSegment
import speechbrain as sb
from speechbrain.dataio.dataio import read_audio
import threading


torch.cuda.is_available()

# Create a function to transcribe audio using the huggingface transformers library

def transcribe_audio_transformers(audio_file, model_name):

    # Load the model and tokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Transcribe

    input_ids = tokenizer.encode(audio_file, return_tensors='pt')
    beam_output = model.generate(input_ids, num_beams=5, num_return_sequences=3, early_stopping=True)
    transcription = tokenizer.decode(beam_output[0], skip_special_tokens=True)

    return transcription




def main(model, english, verbose, energy, pause, dynamic_energy, save_file):
    if save_file==True:
        temp_dir = tempfile.mkdtemp()
        save_path = os.path.join(temp_dir, "temp.wav")

    # there are no english models for large
    if model != "large" and english:
        model = model + ".en"
    audio_model = whisper.load_model(model)

    # load the speech recognizer and set the initial energy threshold and pause threshold
    r = sr.Recognizer()
    r.energy_threshold = energy
    r.pause_threshold = 1
    r.dynamic_energy_threshold = dynamic_energy

    mics = sr.Microphone.list_microphone_names()
    print(mics)
    mic = sr.Microphone(device_index=0, sample_rate=16000)

    with mic as source:
        print("Say something!")
        #st.write("Say something!")
        while True:
            with superman_lock:
                print('superman', superman)
                if not superman:
                    print('not superman')
                    # get and save audio to wav file
                    audio = r.listen(source, timeout=1)
                    print("Got it! Now to recognize it...")
                    if save_file==True:
                        data = io.BytesIO(audio.get_wav_data())
                        audio_clip = AudioSegment.from_file(data)
                        audio_clip.export(save_path, format="wav")
                        audio_data = save_path
                    else:
                        torch_audio = torch.from_numpy(
                            np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
                        audio_data = torch_audio
                        print('audio_data', audio_data)

                    if english:
                        result = audio_model.transcribe(audio_data, language='english')
                    else:
                        result = audio_model.transcribe(audio_data)

                    if not verbose:
                        predicted_text = result["text"]
                        print("You said: " + predicted_text)
                        #st.write("You said: " + predicted_text)
                        signal = read_audio(audio_data)
                        #st.audio(signal, format="audio/wav")
                    else:
                        print(result)
                        #st.write(result)


def loop(model, english, verbose, energy, pause, dynamic_energy, save_file):
    global superman
    global superman_lock
    superman = False
    superman_lock = threading.Lock()
    thread = threading.Thread(target=main, args=(model, english, verbose, energy, pause, dynamic_energy, save_file))
    thread.start()


def endloop():
    global superman
    superman = True





model_options = ["tiny","base","small", "medium", "large"]


st.title("Whisper To Me")

st.write("a demo of the whisper library")

settings, audio_file = st.columns(2)

with settings:
    settings_form = st.form(key="settings_form")
    with settings_form:
        model = settings_form.selectbox("Model", model_options)
        english = settings_form.checkbox("English")
        verbose = settings_form.checkbox("Verbose")
        energy = settings_form.slider("Energy Threshold", 0, 100, 50)
        pause = settings_form.slider("Pause Threshold", 0.0, 10.0, 0.5)
        dynamic_energy = settings_form.checkbox("Dynamic Energy Threshold")
        save_file = settings_form.checkbox("Save File")
        submit = settings_form.form_submit_button("Submit")
        if submit:
            loop(model, english, verbose, energy, pause, dynamic_energy, save_file)
            st.write("Listening...")
            st.write("Press the button below to stop listening")


with audio_file:
    audio_file_form = st.form(key="audio_file_form")
    with audio_file_form:
        audio_file = audio_file_form.file_uploader("Upload Audio File")
        submit = audio_file_form.form_submit_button("Submit")
        if submit:
            if audio_file is not None:
                audio_file = audio_file.read()
                loop(audio_file, english, verbose, energy, pause, dynamic_energy, save_file)
                st.write("Processing...")
                st.write("Press the button below to stop processing")


stop = st.button("Stop Listening")
if stop:
    endloop()
    st.write("Stopped Listening")





