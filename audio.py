import base64
import requests
import os
import sounddevice as sd
import soundfile as sf
from config import GOOGLE_API_KEY, ELEVENLABS_API_KEY, VOICE_ID, SAVE_DIR


class AudioFeedbackPlayer:
    """
    Plays specific musical instrument sounds based on detected color.
    Ensures that the same sound isn't replayed repeatedly unless color changes.
    """

    def __init__(self):
        self.last_color = None
        self.color_sounds = {
            "Red": "sounds/red_guitar.wav",
            "Blue": "sounds/blue_flute.wav",
            "Green": "sounds/green_conga.wav",
            "Yellow": "sounds/yellow_piano.wav",
            "Black": "sounds/black_sax.wav",
        }

    def play_sound(self, color):
        # Stop sound if color is white
        if color == "White":
            sd.stop()
            self.last_color = "White"
            return

        # Avoid replaying the same sound
        if color == self.last_color:
            return

        sound_file = self.color_sounds.get(color)
        if not sound_file or not os.path.exists(sound_file):
            print(f"Sound file for color '{color}' not found.")
            return

        # Update last played color and play associated sound
        self.last_color = color
        data, samplerate = sf.read(sound_file)
        sd.play(data, samplerate, blocking=False)


def speech_to_text(audio_file):
    """
    Converts speech in an audio file to text using Google Speech-to-Text API.
    Supports English (US) and Hindi (India).
    """
    if not os.path.exists(audio_file):
        print("STT Error: Audio file not found.")
        return "Error: No audio file found."

    # Read and encode audio as base64
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    # Construct request payload
    url = f"https://speech.googleapis.com/v1/speech:recognize?key={GOOGLE_API_KEY}"
    payload = {
        "config": {
            "encoding": "MP3",
            "sampleRateHertz": 16000,
            "languageCode": "en-US",
            "alternativeLanguageCodes": ["hi-IN"],
        },
        "audio": {"content": audio_base64},
    }

    # Send request and parse response
    response = requests.post(url, json=payload)
    result = response.json()

    try:
        return result["results"][0]["alternatives"][0]["transcript"]
    except Exception:
        print("STT Error:", result)
        return "Error: Could not transcribe audio."


def text_to_speech(response_text):
    """
    Converts text to speech using ElevenLabs API and saves it as an MP3 file.
    Returns the path to the saved file.
    """
    if not response_text or response_text.strip() == "":
        print("TTS Error: No response text provided.")
        return None

    # Construct request
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
    }
    data = {
        "text": response_text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.8,
            "similarity_boost": 0.85,
            "style": 0.5,
            "use_speaker_boost": True,
        },
    }

    # Send request and save result
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        file_path = os.path.join(SAVE_DIR, "output_audio.mp3")
        with open(file_path, "wb") as f:
            f.write(response.content)
        return file_path
    else:
        print("TTS Error:", response.text)
        return None
