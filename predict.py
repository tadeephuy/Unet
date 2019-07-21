from Unet import Unet
from matplotlib import pyplot as plt

unet = Unet(config_path="model_config.json", weights_path="weights/Best-Epoch26-Loss0.0508.pt")

a = unet.process("D:/Dataset/ICDAR2013/Train_data/111.jpg")

print(a.shape)
plt.imshow(a)
plt.show()