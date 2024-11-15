import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms.functional as F2
from PIL import Image
import numpy as np
import argparse
import json
import clip
from nltk.translate import meteor_score, bleu_score
import matplotlib.pyplot as plt
from imageio import imread
from skimage.transform import resize as imresize
from torchvision import transforms

def predict(img, k=1):
    image = img

    # Encode
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        prev_word_inds = prev_word_inds.long()

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    words = [rev_word_map[ind] for ind in seq]
    sentence = ' '.join(words[1:-1])
    return sentence

device = 'cuda' if torch.cuda.is_available else 'cpu'

clip_model, clip_preprocessor = clip.load("ViT-B/32", device='cuda')

checkpoint = torch.load("model/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar", map_location=str(device))
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

with open("model/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json", 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

reference_list = ['a woman is taking photo',
                  'a man holding a phone',
                  'a man holding a phone',
                  'a bird is sitting on a bird flying near a street',
                  'a group of birds are standing in the water',
                  'a woman is holding a pair of shoes',
                  'a man is watching tv',
                  'a man is standing in a room with a lot of people',
                  'a man is playing with a tennis game',
                  'a group of people sitting down',
                  'a group of people sitting down',
                  'a man is standing in a room with a man on his phone',
                  'a man is standing in a room with a man on his phone',
                  'a man is standing in a room with a man on his phone',
                  'a dog is looking up at the camera',
                  'a woman is taking a picture of a dog',
                  'a person is taking a picture of a person on a cell phone',
                  'a photo of a pair of scissors',
                  'a man holding a toothbrush',
                  'a man holding a toothbrush'
                  ]


# file_name = [3, 4, 5, 7, 9, 10, 20, 24, 25, 26, 27, 28, 31, 35, 36, 44, 61, 74, 77, 79]
file_name = [20]

for idx, file in enumerate(file_name):
    clean_img = (imresize(imread("graybox_attack/" + str(file) + "/orig_" + str(file) + ".png"), (224, 224)).transpose(2, 0, 1)*2)-1
    target_img = (imresize(imread("graybox_attack/" + str(file) + "/target.jpg"), (224, 224)).transpose(2, 0, 1)*2)-1

    processed_clean_image = torch.FloatTensor(clean_img).unsqueeze(0).to(device)
    processed_target_image = torch.FloatTensor(target_img).unsqueeze(0).to(device)

    noise = torch.zeros_like(processed_clean_image, device=device, requires_grad=True)
    optimizer = Adam([noise], lr=0.1)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=30, factor=0.99, cooldown=30, verbose=True)

    with torch.no_grad():
        # x_emb = encoder(processed_target_image).reshape(1, -1)[0]
        x_emb = clip_model.encode_image(processed_target_image)[0].detach()

    cur_losses = []
    for epoch in range(1000):
        optimizer.zero_grad()
        x_adv = (processed_clean_image + noise).cuda()
        x_adv_emb = clip_model.encode_image(x_adv)
        l2_dist = torch.norm((noise).reshape(len(noise), -1), p=2, dim=1)

        targeted_loss = 1 - nn.CosineSimilarity()(x_adv_emb, x_emb).mean()
        # targeted_loss = nn.MSELoss(x_adv_emb, x_emb)
        loss = targeted_loss + 0.1 * l2_dist

        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        noise.data.clamp_(-15/255, 15/255)
        cur_losses.append(loss.data.item())

        if epoch % 100 == 99:
            print(f'Epoch #{epoch + 1} loss: {loss.data[0]:.4f}')

    resize = transforms.Resize((256, 256))

    adv_pred = predict(resize(x_adv))
    print(file, f' After attack:\n\t{adv_pred}')

    clip_model = clip.load("ViT-B/32", device='cuda')[0]
    generated_text = adv_pred
    reference_texts = [reference_list[idx]]
    meteor = meteor_score.meteor_score(reference_texts, generated_text)
    score = bleu_score.sentence_bleu(reference_texts, generated_text)
    generated_features = clip_model.encode_text(clip.tokenize(generated_text).cuda())
    reference_features = clip_model.encode_text(clip.tokenize(reference_texts[0]).cuda())
    cos_sim = F.cosine_similarity(generated_features, reference_features)
    print("The METEOR score is:", meteor, " The BLEU score is: ", score, " The clip cos_sim: ", cos_sim.item())

    clean_pred = predict(resize(processed_clean_image))
    print(file, f' before attack:\n\t{clean_pred}')

    generated_text = clean_pred
    reference_texts = [reference_list[idx]]
    meteor = meteor_score.meteor_score(reference_texts, generated_text)
    score = bleu_score.sentence_bleu(reference_texts, generated_text)
    generated_features = clip_model.encode_text(clip.tokenize(generated_text).cuda())
    reference_features = clip_model.encode_text(clip.tokenize(reference_texts[0]).cuda())
    cos_sim = F.cosine_similarity(generated_features, reference_features)
    print("The METEOR score is:", meteor, " The BLEU score is: ", score, " The clip cos_sim: ", cos_sim.item())


print("1")
