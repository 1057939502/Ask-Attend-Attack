import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from imageio import imread
from skimage.transform import resize as imresize
from PIL import Image
import geatpy as ea
import clip
import time
from scipy.ndimage import zoom
from nltk.translate import meteor_score, bleu_score

def scale_img(x):
    out = (x - x.min()) / (x.max() - x.min())
    return out


def predict(img, k=1):
    img = torch.FloatTensor(img).to(device)
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    # transform = transforms.Compose([normalize])
    # image = transform(img)  # (40, 3, 256, 256)
    image = img


    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
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

class ImageOptimization(ea.Problem):  # Inherited from Problem class.
    def __init__(self, image, target_text):
        name = 'ImageOptimization'  # Problem's name.
        M = 1  # Set the number of objects.
        maxormins = [1]  # All objects are need to be minimized.
        Dim = image.flatten().shape[0]  # Set the dimension of decision variables.
        varTypes = [0] * Dim  # Set the types of decision variables. 0 means continuous while 1 means discrete.
        lb = np.clip( # <<waiting for open source>> )
        ub = np.clip(# <<waiting for open source>>)
        # lb = np.clip(# <<waiting for open source>>)
        # ub = np.clip(# <<waiting for open source>>)
        lbin = [1] * Dim  # Whether the lower boundary is included.
        ubin = [1] * Dim  # Whether the upper boundary is included.
        self.image_shape = image.shape
        self.image = image
        self.adv_image_in_current_step = 0
        self.target_text = target_text
        self.clip_model = clip.load("ViT-B/32", device='cuda')[0]
        self.num = 0
        self.LB = lb
        self.UB = ub
        self.adv_text_features = 0
        self.best_text = 0

        with torch.no_grad():
            target_text_token = clip.tokenize(target_text).cuda()
            target_text_features = self.clip_model.encode_text(target_text_token)
            target_text_features = target_text_features / target_text_features.norm(dim=1, keepdim=True)
            self.target_text_features = target_text_features.detach()

        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        start = time.time()
        self.num = self.num + 1

        x = pop.Phen.reshape((-1,) + self.image_shape)

        predicted_texts = []
        for i in range(NIND):
            predicted_texts.append(predict(x[i], 1))

        target_texts = [self.target_text] * NIND

        loss_1 = # <<waiting for open source>>
        pop.ObjV = loss_1

        end = time.time()
        print(self.num, " time: ", end - start, " cos_sim: ", loss_1.min(), predicted_texts[loss_1.argmin()])

        if self.num % 30 == 1:
            self.best_text = predicted_texts[loss_1.argmin()]

        with open(txt_path, 'a') as file:
            file.write(predicted_texts[loss_1.argmin()] + '\n')
        #     file.write(predicted_texts[0] + '\n')
        #     file.write(predicted_texts[10] + '\n')
        #     file.write(predicted_texts[20] + '\n')
        #     file.write(predicted_texts[30] + '\n')
        #
        # print(predicted_texts[0] + '\n')
        # print(predicted_texts[10] + '\n')
        # print(predicted_texts[20] + '\n')
        # print(predicted_texts[30] + '\n')

        average_curve.append(loss_1.mean())
        best_curve.append(loss_1.min())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

txt_path = 'geatpy_attack/20/dictionary_1.txt'

img = imread("geatpy_attack/20/orig_20.png")
img = imresize(img, (256, 256)).transpose(2, 0, 1)
img = (img * 2) - 1

gradcam = np.load("geatpy_attack/20/orig_20_gradcam.npy")
gradcam = zoom(gradcam, [t/s for t, s in zip((256, 256), (224, 224))], order=1)
gradcam_expand = np.repeat(np.expand_dims(gradcam, axis=0), 3, axis=0)

average_curve = []
best_curve = []
NIND = 40

# target_texts = 'a man is looking at a cell phone in a mirror'
target_texts = "a photo of a parking lot"
problem = ImageOptimization(img, target_texts)

algorithm = ea.soea_DE_rand_1_bin_templet(problem,
                                      ea.Population(Encoding='RI', NIND=NIND),
                                      MAXGEN=10,
                                      logTras=10)

# algorithm.mutOper.F = 0.7
# algorithm.recOper.XOVR = 0.7

# algorithm = ea.soea_EGA_templet(problem,
#                                       ea.Population(Encoding='RI', NIND=NIND),
#                                       MAXGEN=200,
#                                       logTras=10)


res = ea.optimize(algorithm, verbose=False, drawing=1, outputMsg=True, drawLog=True, saveFlag=True, dirName='result')

out_image = np.reshape(res['Vars'], (3, 256, 256))
plt.imshow(scale_img(out_image.transpose((1, 2, 0))))

model_orig_pred = predict(out_image)
print(model_orig_pred)

clip_model = clip.load("ViT-B/32", device='cuda')[0]
generated_text = model_orig_pred
reference_texts = [target_texts]
meteor = meteor_score.meteor_score(reference_texts, generated_text)
score = bleu_score.sentence_bleu(reference_texts, generated_text)
generated_features = clip_model.encode_text(clip.tokenize(generated_text).cuda())
reference_features = clip_model.encode_text(clip.tokenize(reference_texts[0]).cuda())
cos_sim = F.cosine_similarity(generated_features, reference_features)
print("The METEOR score is:", meteor, " The BLEU score is: ", score, " The clip cos_sim: ", cos_sim.item())

noise = img - out_image
print("noise  最大值： ", noise.max(), " 最小值: ", noise.min(), " 平均值: ", noise.mean(), " 总值: ", np.abs(noise).sum(), " 像素平均值: ", np.abs(noise).sum() / (256*256*3))
