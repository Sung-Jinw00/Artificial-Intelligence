import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color

""" Testing np array modifications """
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


""" Testing some maths """
A = np.random.randint(0, 10, [5, 5])
print (A)
print(A.sum(axis=0))
print(A.sum(axis=1))
values, counts = np.unique(A, return_counts=True)
for i, j in zip(values[np.argsort(-counts)], counts[np.argsort(-counts)]):
    print(f'{i} appears {j} time(s)')

data = A
print("\ndata :\n", data, "\n")
data[data % 3 == 0] *= 4
print("data :\n", data, "\n")
mean_val = np.mean(data)
median_val = np.median(data)
var_val = np.var(data)
std_val = np.std(data)
print(f"Mean: {mean_val}, Median: {median_val}, Variance: {var_val}, Std: {std_val}\n")

data2 = data.astype(float)
print("data2 :\n", data2, "\n")
data2[data2 > 5] += 10
print("data2 :\n", data2, "\n")
data2[0] = np.nan
data2[4] = np.nan

mean_nan = np.nanmean(data2)
median_nan = np.nanmedian(data2)
var_nan = np.nanvar(data2)
std_nan = np.nanstd(data2)
print(f"Mean: {mean_nan}, Median: {median_nan}, Variance: {var_nan}, Std: {std_nan}")
