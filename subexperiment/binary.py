import cv2 
import numpy as np
import os
import matplotlib.pyplot as plt



image = cv2.imread('data/train/N25S10_id1286_R_R.png',0)
mask  = cv2.imread('data/high_quality_255/N25S10_id1286_R_R.png',0)

blurred = cv2.GaussianBlur(image, (5, 5), 0)
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)



# 
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 調整核的大小和形狀

# 應用Hat濾波器
hat_filtered = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)


resultmix = cv2.addWeighted(mask, 0.2, hat_filtered, 0.8, 0)

kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

# 連通組件分析
_, labels, stats, centroids = cv2.connectedComponentsWithStats(closing)

# 設定面積和距離閾值
min_area = 1000  # 最小面積閾值
min_distance = 10  # 最小距離閾值

# 繪製結果圖像
result_image = np.zeros_like(image)
for i in range(1, len(stats)):
    if stats[i][4] >= min_area:
        x, y = int(centroids[i][0]), int(centroids[i][1])
        cv2.circle(result_image, (x, y), 5, (255, 255, 255), -1)
        for j in range(1, len(stats)):
            if i != j and stats[j][4] >= min_area:
                distance = np.sqrt((centroids[i][0] - centroids[j][0])**2 + (centroids[i][1] - centroids[j][1])**2)
                if distance < min_distance:
                    cv2.line(result_image, (x, y), (int(centroids[j][0]), int(centroids[j][1])), (255, 255, 255), 2)


hataddkernel = cv2.addWeighted(result_image, 0.4, hat_filtered, 0.6, 0)

plt.subplot(1, 5, 1)
plt.imshow(mask,cmap='gray')
plt.subplot(1,5,2)
plt.imshow(image, cmap= 'gray')
plt.subplot(1, 5, 3)
plt.imshow(result_image,cmap='gray')
plt.subplot(1, 5, 4)
plt.imshow(hat_filtered,cmap='gray')
plt.subplot(1, 5, 5)
plt.imshow(hataddkernel,cmap='gray')

plt.show()
