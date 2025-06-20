import pyttsx3

def pronounce(sentence):
    engine = pyttsx3.init()
    engine.say(sentence)
    engine.runAndWait()

sentence = input("Enter the sentence you want the PC to pronounce: ")
pronounce(sentence)