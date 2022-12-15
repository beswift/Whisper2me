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

session_token = 'eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..08dim9_AUG2e1Yfv.ZNQtykZqa53uOXmnj-zVqevgGQ5i5C9Q4m09-FxO9PuW96H1iGDUFrmgVnohdQNZLGhigXNbJPsnNQJCahA8qMtn1OrkrTc8Fdv0QnGI_vhJGHRzKSXKYN2x4HF-K8x0qxsk4xU1RyF-aYqjs8IuQMtf_rm4lq1NjKSsBijv8IWtDeOFjMruMZ4cbHYS8uV14Rlb-NlD3m-QIFVAXRS4ofLPja9ZfUk8IZh2UlEe_wsKFsO3x7LY9CSw4zI_oevMV1VWbMW9Bfym1dN4BsBX9UwUyVzUTnb8jG3gQgmhIcPCQWbWYCCfKOCaWmC44FWar-nRVEJtbGH_24smIElc8_uHk5v-HKYkxwf3PzpJnwowobt1VfRC0tbrRtRRJPW_uqKL63ZIVuoy6M-kG2lF_kT91LyCY-Lwi_IlpyOStccOWbUVUa-8DFBBj3udtinEzSg1UPu1ZcyLvSaDJRialyKpMYHixcao_Inl-LFOKpDkHtP_j-GNA61M6dnhuwe5k1AJ0P2BWypi5K3I7516qcrJLq2RF6OdkOdZVD3Ykme56P0eJPvdwSc7RFUi_nMUjmgS8PjP7Bz85oqazRF90SsWFzawgqbg_zB03csXjIXL9s5o6fmI72qGs31FByBoQK2mgArXxXyJMl9mfKMUNYEgR8RRp8bi7MQjH_L1eNtnm8spylGhV7Rpi2k4H6AFao0wWUFNGMXJZ8w920rM0f3ZbF8GrJdarpIzGsSdkUJ5z0YECiwzC_KB_Jwon-q1hnBIwJCyCg4HpSqgu31jFcZ7vO6YiyELqqWRx5SCLqeK3JHAMP2gYnloyC-QQFPO1UUpSMMdilFNpOpbpAGjKy42R0wIYxaRq2dW9ybMRviyNcbPtgyr8Lj363AEQopSbTGcHflMjgkxc5snhUZ4w_r-cUDMC0E2cHXH0TBsN7ZY4fA3_cvaJ-141Pcqe71Y5gXp-WmVB_9DtuLONYXzsjSm9vSutGc7A9sboZc0cGBMBug2r5V_hh_xcBqsqleWf7xB2fYsc9ruhKvbM-WDYUy0tOA_a2zEJ8lDln7bSRqTfbW5PlAgTotuxR9K2ewBFwnTjZ_ZiAguZSgpTUrscZkjsg3I3vq6yIKegqa0KphG0GkW5ykAzH4CXBERlqBiHKIUpRPTd1h8jj5dgDTgqVy20TNMNfMlKhSwNPNbeir8EwCX8aHJd5rCt312tPB3bDDSEisGnKuNGIJRQ7bbf1zZ_c1nhPnwi6AGzux6x5DkJKmXBUd0PECBcxYexKkdfjAclehzMNI3zbDL7_Gsd1xbT9pDyfuIZDQgKHhXktfGS0UwtwNYaQnXKqbD5oTwGXLg6jc0tn7bWD63ZbHnU5Xge8o8p7RJXeGJ7elgLhGVNewrzAQkcMrNZvoy7dyWXVJoYv7ZXBwAFz_lKC66APOEFTr16SzU393RD9mpf9jHo3nTyT9vRXDGqbQY8ONdvjVhiuunSt0RhjNMdvMOgkMTQBEfOjk9RTNRVTVXoj4eDsC7pc8XmtxGZ2UINoHBg1-jI-0_DIlLThAn94oU4jqakIbeKHa81zHF0tBcC-j28F4q2pCv9Rj1txKiGCQimcudKTbYtEk34GM5K6otQNr-lcyDUPg6TqeNqlDK6GlJtpvbtEB_nRfLlDENwI-QY9gPgE7WnloBduESzne5e5BnNbTOBRswDjZCDJcWdGL-S9I6El691-pE8zcVg6-VHaITxQEQ7Iw-gESw5xdFWWZMMk3_LISL5kLA0jj5vZ2ZmJZqLp-MLt7S6XrI2OzJi7aaM1tjIuytgGnXXoua6ubqXYYg1k3zpAItVsRSOxyqSExw6wJ1FsbatlAUfBj8j5VnA135iO22ACDzKhCusGmw1gvyflPaXJgQEUBktLwWAAldDs8TupmPOXnLxLYAdWlrLGDoumQQ29aU8s2gVhmG2Mj7aNPdo6S_Iq0tH4rlL8XBRr68Rht4m6YChUv418M6nI1o2u34KUoLRAnskHR6wwv2IQDyFR6pKIYOBvVVBKrPXU3poPcOUXk4p6VjL_ymd63GNVVTf7rHwV2sM056hcx7oVxzOyGHn5PRdNf2auKGXvb1_NM_rNVoDT-fPFqMn10-4lDrMrF-6G2Zpxzi4pjR-DKxntBZf3eHQwztrZyLqahZ4XZ1Hx0yOujQ6Dv7OQCeJxFwxtoiSw3QS-wmErKEh1ecaDzkQHKWH2a4IDR5K8HmSDQTZ4zs4mdP0oTiYpjs9shDUNZo1BVCjADuC0W_I868xuEwLYkyxZHgS9H5zjhFVwN09RJAfxqqy61ovxuRHgDqlIixgYk2lrkWiuhuF7pVXDuZExmJrnz55Zp6vMfT0HHN9yzzzC3RU4Pi.s20_vvfB-PxFV6YDAbqFTg'
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
print(said)
speak(f'you said:{said}',filename='said')

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

