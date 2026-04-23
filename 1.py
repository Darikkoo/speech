import speech_recognition as sr
import pyttsx3
import os
import sys
import webbrowser
import datetime
import subprocess
import requests

engine = pyttsx3.init()

def talk(words):
    print(words)
    engine.say(words)
    engine.runAndWait()


def command():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print("Say...")
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source, duration=1)
        audio = r.listen(source)

    try:
        zadanie = r.recognize_google(audio, language="ru-RU").lower()
        print("You say:", zadanie)
    except sr.UnknownValueError:
        talk("I don't understand")
        return command()

    return zadanie, audio

def get_time():
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M")
    talk(f"Текущее время: {current_time}")


def open_application(app_name):
    try:
        if app_name == "notepad":
            subprocess.Popen(["notepad.exe"])
            talk("Opening Notepad")

        elif app_name == "calculator":
            subprocess.Popen(["calc.exe"])
            talk("Opening Calculator")

        else:
            talk("Sorry, I don't know this application.")

    except Exception as e:
        talk(f"Failed to open {app_name}. Error: {e}")


def makeSomething(zadanie, audio_data):

    if "открыть сайт" in zadanie:
        webbrowser.open("https://itproger.com")
        talk("Opening website")

    elif "stop" in zadanie:
        talk("Goodbye")
        sys.exit()

    elif "имя" in zadanie:
        talk("Менің атым - Voice Assistant")

    elif "время" in zadanie:
        get_time()

    elif "открыть приложение" in zadanie:
        talk("Which app?")
        app = command()[0]
        open_application(app)

talk("Assistant started")

while True:
    zadanie, audio_data = command()
    makeSomething(zadanie, audio_data)
