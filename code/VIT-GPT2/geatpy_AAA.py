import numpy as np
import geatpy as ea
from PIL import Image
from utils import load_model, predict
import torch
import clip
import torch.nn.functional as F
import matplotlib.pyplot as plt
import nltk
import time

from nltk.translate import meteor_score, bleu_score

def scale_img(x):
    out = (x - x.min()) / (x.max() - x.min())
    return out

def exponential_moving_average(data, alpha):
    """
    计算指数滑动平均
    :param data: 输入的numpy数组
    :param alpha: 平滑因子，范围在0到1之间
    :return: 指数滑动平均后的numpy数组
    """
    ema_data = np.zeros_like(data)
    ema_data[0] = data[0]
    for t in range(1, len(data)):
        ema_data[t] = alpha * data[t] + (1 - alpha) * ema_data[t - 1]
    return ema_data

def non_increasing(data):
    """
    将数组转换为非递增数组
    :param data: 输入的numpy数组
    :return: 非递增的numpy数组
    """
    for i in range(1, len(data)):
        if data[i] > data[i - 1]:
            data[i] = data[i - 1]
    return data


class ImageOptimization(ea.Problem):  # Inherited from Problem class.
    def __init__(self, image, target_text, model):
        name = 'ImageOptimization'  # Problem's name.
        M = 1  # Set the number of objects.
        maxormins = [1]  # All objects are need to be minimized.
        Dim = image.flatten().shape[0]  # Set the dimension of decision variables.
        varTypes = [0] * Dim  # Set the types of decision variables. 0 means continuous while 1 means discrete.
        lb = np.clip(# <<wait for open source>>)
        ub = np.clip(# <<wait for open source>>)
        lbin = [1] * Dim  # Whether the lower boundary is included.
        ubin = [1] * Dim  # Whether the upper boundary is included.
        self.image_shape = image.shape
        self.image = image
        self.adv_image_in_current_step = 0
        self.target_text = target_text
        self.model = model
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

    def push_1(self, predicted_texts, x):
        adv_vit_text_token_in_current_step = clip.tokenize(predicted_texts).cuda()
        adv_vit_text_features_in_current_step = self.clip_model.encode_text(adv_vit_text_token_in_current_step)
        adv_vit_text_features_in_current_step = adv_vit_text_features_in_current_step / adv_vit_text_features_in_current_step.norm(dim=1, keepdim=True)
        adv_vit_text_features_in_current_step = adv_vit_text_features_in_current_step.detach()
        self.adv_text_features = adv_vit_text_features_in_current_step
        self.adv_image_in_current_step = x
        torch.cuda.empty_cache()

    def push_2(self, text_of_perturbed_imgs, x):
        perturb_text_token = clip.tokenize(text_of_perturbed_imgs).cuda()
        perturb_text_features = self.clip_model.encode_text(perturb_text_token)
        perturb_text_features = perturb_text_features / perturb_text_features.norm(dim=1, keepdim=True)
        perturb_text_features = perturb_text_features.detach()
        coefficient = torch.sum((perturb_text_features - self.adv_text_features) * self.target_text_features, dim=-1)
        coefficient = coefficient.reshape(NIND, 1, 1, 1)
        query_noise = torch.from_numpy(x - self.adv_image_in_current_step).cuda()
        pseudo_gradient = coefficient * query_noise / 0.5
        pseudo_gradient = pseudo_gradient.mean(0)
        return pseudo_gradient

    def aimFunc(self, pop):
        start = time.time()
        self.num = self.num + 1

        x = pop.Phen.reshape((-1,) + self.image_shape)
        predicted_texts = predict('vit-gpt2', model, tokenizer, image_processor,
                                  torch.from_numpy(x).clone().to('cuda').float())

        target_texts = [self.target_text] * NIND

        # if self.num <= 50:
        #     target_texts = [self.target_text] * NIND
        # else:
        #     target_texts = [self.best_text] * NIND

        # if self.num % 2 == 1:
        #     self.push_1(predicted_texts, x)
        # else:
        #     pseudo_gradient = self.push_2(predicted_texts, x)
        #     x = torch.clamp(torch.from_numpy(x).clone().to('cuda').float() + pseudo_gradient, -1.0, 1.0)
        #     pop.Phen = x.cpu().detach().numpy().reshape((NIND, -1))
        #     predicted_texts = predict('vit-gpt2', model, tokenizer, image_processor, x.clone().to('cuda').float())
        #     print("pseudo_gradient: ", pseudo_gradient.max().item(), pseudo_gradient.mean().item(), pseudo_gradient.min().item())

        loss_1 = # <<wait for open source>>
        loss_1 = np.reshape(loss_1, (-1, 1))
        pop.ObjV = loss_1

        # noise = x - self.image
        # pop.CV = np.abs(noise).max((1, 2, 3)).reshape((-1, 1)) - 1
        # print(self.num, " time: ", end - start, " cos_sim: ", loss_1.min(), predicted_texts[loss_1.argmin()], "噪音最大值: ", noise.max())

        end = time.time()
        print(self.num, " time: ", end - start, " cos_sim: ", loss_1.min(), predicted_texts[loss_1.argmin()])

        if self.num % 30 == 1:
            self.best_text = predicted_texts[loss_1.argmin()]

        with open(txt_path, 'a') as file:
            file.write(predicted_texts[loss_1.argmin()] + '\n')
        average_curve.append(loss_1.mean())
        best_curve.append(loss_1.min())


image_processor, tokenizer, model, encoder, image_mean, image_std = load_model(model_name='vit-gpt2')

img = Image.open("specific/geatpy_attack/20/orig_20.png")
processed_image = image_processor(img, return_tensors="pt")['pixel_values']
gradcam = np.load("specific/geatpy_attack/20/orig_20_gradcam.npy")
gradcam_expand = np.repeat(np.expand_dims(gradcam, axis=0), 3, axis=0)
txt_path = 'specific/geatpy_attack/20/dictionary_1.txt'

average_curve = []
best_curve = []

NIND = 40

target_texts = 'a woman with a cowboy hat standing next to a black horse'
# target_texts = 'a cat is stting on the ground'
problem = ImageOptimization(processed_image[0].numpy(), target_texts, model)

algorithm = ea.soea_DE_best_1_bin_templet(problem,
                                      ea.Population(Encoding='RI', NIND=NIND),
                                      MAXGEN=200,
                                      logTras=10)

# algorithm = ea.soea_DE_currentToBest_1_bin_templet(problem,
#                                       ea.Population(Encoding='RI', NIND=NIND),
#                                       MAXGEN=200,
#                                       logTras=10)

# algorithm = ea.soea_DE_targetToBest_1_bin_templet(problem,
#                                       ea.Population(Encoding='RI', NIND=NIND),
#                                       MAXGEN=200,
#                                       logTras=10)

# algorithm = ea.soea_studGA_templet(problem,
#                                       ea.Population(Encoding='RI', NIND=NIND),
#                                       MAXGEN=40,
#                                       logTras=10)


algorithm.mutOper.F = 0.7
algorithm.recOper.XOVR = 0.7

res = ea.optimize(algorithm, verbose=False, drawing=1, outputMsg=True, drawLog=True, saveFlag=True, dirName='result')

out_image = np.reshape(res['Vars'], (1, 3, 224, 224))
# plt.imshow(scale_img(out_image[0].transpose((1, 2, 0))))

model_orig_pred = predict('vit-gpt2', model, tokenizer, image_processor, torch.from_numpy(out_image).to('cuda'))
print(model_orig_pred)

clip_model = clip.load("ViT-B/32", device='cuda')[0]
generated_text = model_orig_pred[0]
reference_texts = [target_texts]
meteor = meteor_score.meteor_score(reference_texts, generated_text)
score = bleu_score.sentence_bleu(reference_texts, generated_text)
generated_features = clip_model.encode_text(clip.tokenize(generated_text).cuda())
reference_features = clip_model.encode_text(clip.tokenize(reference_texts[0]).cuda())
cos_sim = F.cosine_similarity(generated_features, reference_features)
print("The METEOR score is:", meteor, " The BLEU score is: ", score, " The clip cos_sim: ", cos_sim.item())

noise = out_image - processed_image[0].numpy()
print("noise  最大值： ", noise.max(), " 最小值: ", noise.min(), " 平均值: ", noise.mean(), " 总值: ", np.abs(noise).sum(), " 像素平均值: ", np.abs(noise).sum() / (224*224*3))

plt.imshow(scale_img(out_image[0].transpose((1, 2, 0))))
plt.show()
print("1")