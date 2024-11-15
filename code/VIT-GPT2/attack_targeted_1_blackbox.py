import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms.functional as F2
from PIL import Image
import numpy as np
import argparse

import clip

from utils import predict
from utils import load_model
from utils import load_dataset
from utils import save_img_and_text

device = 'cuda' if torch.cuda.is_available else 'cpu'

clip_model, clip_preprocessor = clip.load("ViT-B/32", device='cuda')
image_processor, tokenizer, model, encoder, image_mean, image_std = load_model(model_name="vit-gpt2")

file_name = [3, 4, 5, 7, 9, 10, 20, 24, 25, 26, 27, 28, 31, 35, 36, 44, 61, 74, 77, 79]

for file in file_name:
    clean_img = Image.open("specific/graybox_attack/" + str(file) + "/orig_" + str(file) + ".png")
    target_img = Image.open("specific/graybox_attack/" + str(file) + "/target.jpg")

    processed_clean_image = image_processor(clean_img, return_tensors="pt")['pixel_values']
    processed_target_image = image_processor(target_img, return_tensors="pt")['pixel_values']

    noise = torch.zeros_like(processed_clean_image, device=device, requires_grad=True)
    optimizer = Adam([noise], lr=0.01)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=30, factor=0.1, cooldown=30, verbose=True)

    with torch.no_grad():
        # x_emb = encoder(processed_target_image.cuda())[1][0].detach()
        x_emb = clip_model.encode_image(processed_target_image.cuda())[0].detach()

    cur_losses = []
    for epoch in range(300):
        optimizer.zero_grad()
        x_adv = (processed_clean_image.cuda() + noise).cuda()
        # x_adv_emb = encoder(x_adv)[1]
        x_adv_emb = clip_model.encode_image(x_adv)
        l2_dist = torch.norm((noise).view(len(noise), -1), p=2, dim=1)

        targeted_loss = 1 - nn.CosineSimilarity()(x_adv_emb, x_emb).mean()
        loss = targeted_loss + 0.1 * l2_dist

        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        noise.data.clamp_(-30/255, 30/255)
        cur_losses.append(loss.data.item())

        # if epoch % 100 == 99:
        #     print(f'Epoch #{epoch + 1} loss: {loss.data[0]:.4f}')

    clean_pred = predict('vit-gpt2', model, tokenizer, image_processor, processed_clean_image.cuda())
    adv_pred = predict('vit-gpt2', model, tokenizer, image_processor, x_adv)
    print(file, f' After attack:\n\t{adv_pred}')
    print("clean pred: ", clean_pred)



print("1")
