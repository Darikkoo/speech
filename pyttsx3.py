import cv2
import pytesseract
import pyttsx3
import speech_recognition as sr
import sys

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

engine = pyttsx3.init()

def talk(words):
    print(words)
    engine.say(words)
    engine.runAndWait()

def command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something...")
        r.adjust_for_ambient_noise(source, duration=1)
        audio = r.listen(source)

    try:
        zadanie = r.recognize_google(audio, language="ru-RU").lower()
        print("You said:", zadanie)
    except sr.UnknownValueError:
        talk("I didn't understand, please repeat.")
        return command()

    return zadanie

def find_word_in_image(target_word):
    img = cv2.imread('proverb5.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    config = r'--oem 3 --psm 6'
    recognized_text = pytesseract.image_to_string(img, config=config)
    print("Распознанный текст:", recognized_text)

    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    word_occurrences = [
        i for i, word in enumerate(data["text"])
        if word.lower() == target_word.lower()
    ]

    image_copy = img.copy()

    for occ in word_occurrences:
        w = data["width"][occ]
        h = data["height"][occ]
        l = data["left"][occ]
        t = data["top"][occ]
        cv2.rectangle(image_copy, (l, t), (l + w, t + h), (255, 0, 0), 2)

    cv2.imshow("Detected Words", image_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    talk(recognized_text)

def main():
    while True:
        zadanie = command()

        if 'выход' in zadanie:
            talk("Goodbye!")
            sys.exit()

        elif 'поиск' in zadanie:
            talk("What word would you like to find?")
            word_to_find = command()
            find_word_in_image(word_to_find)

if __name__ == "__main__":
    main()
