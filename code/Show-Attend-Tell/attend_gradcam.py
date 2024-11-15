import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from torchvision.models import resnet50, vgg16, densenet161, vgg19, resnet18
from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50, deeplabv3_resnet101
from torchvision.models.detection import  fasterrcnn_resnet50_fpn
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
import cv2
import torch
from PIL import Image
from utils import *
import torch.optim as optim
from torchvision import transforms
from torch.nn import functional as F
import requests


# img_path = "yolov3_spp_origin/data/my_yolo_dataset_VOC2012/train/images/2008_000074.jpg"
# img_path = "./compare_image/both.png"
# img_path = "./compare_image/2008_002778.jpg"
# img_path = "./compare_image/000000280779.jpg"
# img_path = "./compare_image/000000000785.jpg"
# img_path = "./compare_image/2008_002760.jpg"
img_path = "./compare_image/orig_79.png"

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = Image.open(img_path)
input_tensor = preprocess(img).unsqueeze(0)
print(input_tensor.shape)



# model = densenet161(pretrained=True)
# target_layers = model.features

# model = vgg16(pretrained=True)
# target_layers = [model.features]
# # #
# # # model = resnet50(pretrained=True)
# # # target_layers = [model.layer4]
# #
# #
# model = model.eval()
# cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
# grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(281)])[0]
# visualization = show_cam_on_image(np.array(img).astype(np.float32) / 255.0, grayscale_cam, use_rgb=True)
# plt.imshow(visualization)
# plt.show()





print("-------------")
#
# model = deeplabv3_resnet50(pretrained=True, progress=False)
# #model = deeplabv3_resnet101(pretrained=True, progress=False)
# #model = fcn_resnet50(pretrained=True, progress=False)
#
# model = model.eval()
# if torch.cuda.is_available():
#     model = model.cuda()
#     input_tensor = input_tensor.cuda()
#
# class SegmentationModelOutputWrapper(torch.nn.Module):
#     def __init__(self, model):
#         super(SegmentationModelOutputWrapper, self).__init__()
#         self.model = model
#
#     def forward(self, x):
#         return self.model(x)["out"]
#
# model = SegmentationModelOutputWrapper(model)
# output = model(input_tensor)
# normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()
# sem_classes = [
#     '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
#     'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
#     'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
# ]
# sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
#
# car_category = sem_class_to_idx["horse"]
# car_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
# car_mask_uint8 = 255 * np.uint8(car_mask == car_category)
# car_mask_float = np.float32(car_mask == car_category)
#
# both_images = np.hstack((img, np.repeat(car_mask_uint8[:, :, None], 3, axis=-1)))
# Image.fromarray(both_images)
#
#
# class SemanticSegmentationTarget:
#     def __init__(self, category, mask):
#         self.category = category
#         self.mask = torch.from_numpy(mask)
#         if torch.cuda.is_available():
#             self.mask = self.mask.cuda()
#
#     def __call__(self, model_output):
#         return (model_output[self.category, :, :] * self.mask).sum()
#
#
# target_layers = [model.model.backbone.layer4]
# targets = [SemanticSegmentationTarget(car_category, car_mask_float)]
# with GradCAM(model=model,
#              target_layers=target_layers,
#              use_cuda=torch.cuda.is_available()) as cam:
#     grayscale_cam = cam(input_tensor=input_tensor,
#                         targets=targets)[0, :]
#     visualization = show_cam_on_image(np.array(img).astype(np.float32) / 255.0, grayscale_cam, use_rgb=True)
#
# plt.imshow(visualization)
# plt.show()

print("------------")

class GradCAM_git(object):
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

    def forward_hook(self, module, input, output):
        # Save the activations of the target layer
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        # Save the gradients of the target layer
        self.gradients = grad_output[0]

    def register_hook(self):
        # Register forward and backward hooks to the target layer
        self.forward_handler = self.target_layer.register_forward_hook(self.forward_hook)
        self.backward_handler = self.target_layer.register_backward_hook(self.backward_hook)

    def get_heatmap(self, class_idx, input_tensor):
        # Get the Grad-CAM heatmap for a specific class
        # class_idx: the index of the target class
        # Return: the Grad-CAM heatmap (torch.Tensor)

        # Get the mean of the gradients along the spatial dimensions
        weights = torch.mean(self.gradients, dim=[2, 3])

        # Multiply each activation channel by the corresponding weight
        weighted_activations = weights[:, :, None, None] * self.activations

        # Sum up the weighted activation channels
        heatmap = torch.sum(weighted_activations, dim=1)

        # Normalize and interpolate the heatmap
        heatmap = F.relu(heatmap) / (torch.max(heatmap) + 1e-8)
        heatmap = F.interpolate(heatmap.unsqueeze(1), size=input_tensor.size()[2:], mode='bilinear', align_corners=False).squeeze()

        return heatmap

    def __call__(self, input_tensor, class_idx=None):
        # Compute and return the Grad-CAM heatmap for a specific class
        # input_tensor: the input tensor to the model (torch.Tensor)
        # class_idx: the index of the target class (int)
        # Return: the Grad-CAM heatmap (torch.Tensor)

        # Register hooks to the target layer
        self.register_hook()

        # Forward pass and get the output of the model
        output = self.model(input_tensor)

        if class_idx == None:
            class_idx = output.argmax()

        # Zero out gradients from previous iteration
        self.model.zero_grad()

        # Backward pass with respect to the target class
        output[0][class_idx].backward(retain_graph=True)

        # Get and return the Grad-CAM heatmap
        heatmap = self.get_heatmap(class_idx, input_tensor)

        return heatmap, class_idx

# model = resnet50(pretrained=True)
# gradcam_git = GradCAM_git(model=model, target_layer=model.layer4)

model = densenet161(pretrained=True)
# model = vgg16(pretrained=True)
gradcam_git = GradCAM_git(model=model, target_layer=model.features[-1])

grayscale_cam, class_index = gradcam_git(input_tensor=input_tensor, class_idx=523)
visualization = show_cam_on_image(np.array(img).astype(np.float32) / 255.0, grayscale_cam.detach().numpy(), use_rgb=True)
plt.imshow(visualization)
plt.show()
print("---------")



# def predict(input_tensor, model, device, detection_threshold):
#     outputs = model(input_tensor)
#     pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
#     pred_labels = outputs[0]['labels'].cpu().numpy()
#     pred_scores = outputs[0]['scores'].detach().cpu().numpy()
#     pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
#
#     boxes, classes, labels, indices = [], [], [], []
#     for index in range(len(pred_scores)):
#         if pred_scores[index] >= detection_threshold:
#             boxes.append(pred_bboxes[index].astype(np.int32))
#             classes.append(pred_classes[index])
#             labels.append(pred_labels[index])
#             indices.append(index)
#     boxes = np.int32(boxes)
#     return boxes, classes, labels, indices
#
#
# def draw_boxes(boxes, labels, classes, image):
#     for i, box in enumerate(boxes):
#         color = COLORS[labels[i]]
#         cv2.rectangle(
#             image,
#             (int(box[0]), int(box[1])),
#             (int(box[2]), int(box[3])),
#             color, 2
#         )
#         cv2.putText(image, classes[i], (int(box[0]), int(box[1] - 5)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
#                     lineType=cv2.LINE_AA)
#     return image
#
# def fasterrcnn_reshape_transform(x):
#     target_size = x['pool'].size()[-2 : ]
#     activations = []
#     for key, value in x.items():
#         activations.append(torch.nn.functional.interpolate(torch.abs(value), target_size, mode='bilinear'))
#     activations = torch.cat(activations, axis=1)
#     return activations
#
# coco_names = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
#               'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
#               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
#               'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
#               'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
#               'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
#               'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
#               'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
#               'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#               'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
#               'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
#               'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
#               'scissors', 'teddy bear', 'hair drier', 'toothbrush']
#
# # This will help us create a different color for each class
# COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))
# device = torch.device('cpu')
#
#
# image = np.array(Image.open(img_path))
# image_float_np = np.float32(image) / 255
# # define the torchvision image transforms
# transform = torchvision.transforms.Compose([
#     torchvision.transforms.ToTensor(),
# ])
#
# input_tensor = transform(image)
#
# input_tensor = input_tensor.to(device)
# # Add a batch dimension:
# input_tensor = input_tensor.unsqueeze(0)
#
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# # model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
# model.eval().to(device)
#
# # Run the model and display the detections
# boxes, classes, labels, indices = predict(input_tensor, model, device, 0.9)
# image = draw_boxes(boxes, labels, classes, image)
#
# target_layers = [model.backbone]
# targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]
# cam = EigenCAM(model,
#                target_layers,
#                use_cuda=torch.cuda.is_available(),
#                reshape_transform=fasterrcnn_reshape_transform)
# grayscale_cam = cam(input_tensor, targets=targets)[0]
# cam_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)
# image_with_bounding_boxes = draw_boxes(boxes, labels, classes, cam_image)
# plt.imshow(image_with_bounding_boxes)
# plt.show()
