from collections import Counter
import string

with open('geatpy_attack_image/27/ElephantAndSelfie_200_nogradcam.txt', 'r') as file:
    content = file.read()

words = content.split()
word_frequency = Counter(words)

for word, frequency in word_frequency.items():
    print(f"{word}: {frequency}")

