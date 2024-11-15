from collections import Counter
import string
import clip
import torch.nn.functional as F

clip_model = clip.load("ViT-B/32", device='cuda')[0]

with open('F:/python/pycharm/a-PyTorch-Tutorial-to-Image-Captioning-master/geatpy_attack/61/dictionary_a parking lot 30.txt', 'r') as file:
    content = file.read()

words = content.split()
word_frequency = Counter(words)

num = 0
average_clip = 0
for word, frequency in word_frequency.items():
    print(num, f"{word}: {frequency}")

    generated_text = word
    reference_texts = ["a photo of a parking lot"]
    generated_features = clip_model.encode_text(clip.tokenize(generated_text).cuda())
    reference_features = clip_model.encode_text(clip.tokenize(reference_texts[0]).cuda())
    cos_sim = F.cosine_similarity(generated_features, reference_features)
    print("clip cos_sim: ", cos_sim.item())
    average_clip = average_clip + cos_sim

    num = num + 1

print(" clip average: ", (average_clip / num).item())

print("1")
