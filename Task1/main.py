import numpy as np
import cv2

import torch
import torchvision
from torchvision.models import ResNet50_Weights

avgpool_features = None


def get_features(module, inputs, output):
    global avgpool_features
    avgpool_features = output


model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.layer4.register_forward_hook(get_features)
model.eval()


transform_pipe = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


with open('imagenet1000_clsid_to_human.txt') as file:
    label = [line.strip() for line in file.readlines()]

W = model.fc.weight

cap = cv2.VideoCapture('source.mp4')
while cap.isOpened():
    image = cap.read()[1]
    transformed_image = transform_pipe(image)
    transformed_image = transformed_image[None, :, :, :]

    prob = torch.nn.functional.softmax(model(transformed_image), dim=1)
    index = torch.argmax(prob)

    result = torch.zeros(1, 7, 7)
    for i in range(2048):
        result += W[index, i] * avgpool_features[0, i, :, :]

    heat_map = torchvision.transforms.Resize(size=(360, 640))(result)
    heat_map = heat_map[0].cpu().detach().numpy()
    heat_map = heat_map / heat_map.max() * 255
    heat_map = heat_map.astype(np.uint8)
    heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)

    res = cv2.addWeighted(image, 0.5, heat_map, 0.5, 0)

    cv2.putText(res, label[index][6:-2], (30, 30), cv2.FONT_ITALIC, 0.5, (255, 255, 255))
    cv2.imshow('video', res)
    cv2.waitKey(1)
