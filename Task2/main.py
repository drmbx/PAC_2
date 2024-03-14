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
print(template_features.size())

template_features_flat = torch.flatten(template_features, start_dim=2)
image_features_flat = torch.flatten(image_features, start_dim=2)

# Нормализуем признаки для вычисления косинусного сходства
template_features_norm = F.normalize(template_features_flat, p=2, dim=2)
image_features_norm = F.normalize(image_features_flat, p=2, dim=2)

# Усреднение значений признаков в пределах каждого канала признаков
template_features_norm_mean = torch.mean(template_features_norm, dim=2, keepdim=True)
image_features_norm_mean = torch.mean(image_features_norm, dim=2)

# Вычисляем косинусное сходство с помощью скалярного произведения
cos_sim = torch.matmul(template_features_norm_mean, image_features_norm_mean.permute(0, 1, 3, 2))

# Преобразуем размерности для визуализации тепловой карты
cos_sim = cos_sim.squeeze().cpu().numpy()

print("Размерность тензора cos_sim:", cos_sim.shape)

# Визуализируем тепловую карту
plt.imshow(cos_sim, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()
