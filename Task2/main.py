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

picture_num = 3

print(np.shape(images[1]))
template = images[1][408:474, 230:340]
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

print(model)

input_tensor = preprocess(template)
input_batch = input_tensor.unsqueeze(0)
model(input_batch)
template_features = layer4_features
print(template_features.size())

input_tensor = preprocess(images[picture_num])
input_batch = input_tensor.unsqueeze(0)
model(input_batch)
image_features = layer4_features
print(image_features.size())

heat_map = F.conv2d(image_features, template_features)
heat_map = heat_map.squeeze()
heat_map = heat_map - heat_map.min()
heat_map = ((heat_map / heat_map.max()) * 256).byte()
heat_map = heat_map.squeeze().cpu().numpy()
image_height, image_width = images[picture_num].shape[:2]
heat_map = cv2.resize(heat_map, (image_width, image_height))
heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
res = cv2.addWeighted(images[picture_num], 0.5, heat_map, 0.5, 0)
plt.imshow(res)
plt.show()
