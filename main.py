""" Testing np array modifications """
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color

face = data.astronaut()
gray = color.rgb2gray(face)
plt.imshow(gray, cmap="gray")
plt.axis("off")
plt.show()
print(gray.shape)
height = gray.shape[0]
width = gray.shape[1]
zoom_h = int(height * 1/8)
zoom_w = int(width * 1/8)
gray = gray[zoom_h:(height - zoom_h), zoom_w:(width - zoom_w)]
g_min = np.min(gray)
g_max = np.max(gray)
gray[gray > g_max * 0.8] = g_max
gray[gray < g_min + (g_max-g_min) * 0.2] = g_min
plt.imshow(gray, cmap="gray")
plt.axis("off")
plt.show()
print(gray.shape)