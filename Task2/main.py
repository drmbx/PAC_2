import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn.functional as F

sources = ["A_B_C.jpg", "Ab_C.jpg", "abc.png", "abc.jpg"]
images = []
for source in sources:
    images.append(cv2.cvtColor(cv2.imread(source), cv2.COLOR_BGR2RGB))

print(np.shape(images[1]))
template = images[1][408:484, 210:340]
print(np.shape(template))
plt.imshow(template)
plt.show()

model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
layer4_features = None


def get_features(module, inputs, output):
    global layer4_features
    layer4_features = output


model.layer4.register_forward_hook(get_features)
model.eval()

preprocess = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


input_tensor = preprocess(template)
input_batch = input_tensor.unsqueeze(0)
model(input_batch)
template_features = layer4_features
print(template_features)
print(template_features.size())

input_tensor = preprocess(images[0])
input_batch = input_tensor.unsqueeze(0)
model(input_batch)
image_features = layer4_features
print(image_features)
print(image_features.size())

torch.nn.Conv2d(2048, 1, (3, 5))()
