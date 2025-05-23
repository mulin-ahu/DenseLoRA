import matplotlib.pyplot as plt
import numpy as np

# 创建一个 10x10 的二维数组
data = np.random.rand(10, 10)

# 显示图像，使用不同的颜色映射
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(data, cmap='viridis')
plt.colorbar()
plt.title("Viridis")

plt.subplot(1, 3, 2)
plt.imshow(data, cmap='hot')
plt.colorbar()
plt.title("Hot")

plt.subplot(1, 3, 3)
plt.imshow(data, cmap='gray')
plt.colorbar()
plt.title("Gray")

plt.show()