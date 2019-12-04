# -*- coding: utf-8 -*-
# @Time    : 2019/11/30 15:28
# @Author  : Jeff Wang
# @Email   : jeffwang987@163.com   OR    wangxiaofeng2020@ia.ac.cn
# @Software: PyCharm
"""
本次作业：实现二维的快速中值滤波。
论文参考：A Fast Two-Dimensional Median Filtering Algorithm
作业感悟：该算法比较古老，如果单纯从Pixel角度实现，还是较慢。 OpenCV的中值滤波基于矩阵维度操作，且有各种优化处理，速度比该算法快不少。
"""
import cv2  # 整个code用到OpenCV的地方只有：读取图像+显示图像
import imutils  # 其他的代码全部通过Pixel层面实现，放在imutils.py中
import numpy as np
import time

"""给原图添加噪声"""
image = cv2.imread("Lenna.jpg")
utils = imutils.Utils()
gray = utils.rgb2gray(image)
blur_sp = utils.sp_noise(gray, prob=0.01)

"""执行快速中值滤波算法"""
start = time.time()
output = np.copy(blur_sp)
kernel_size = (3, 3)
task1 = imutils.FastMedianBlur(blur_sp, kernel_size)
for i in range(blur_sp.shape[0]-kernel_size[0]):
    task1.hist2zero()
    window_start = blur_sp[i:i+kernel_size[0], 0:kernel_size[1]]
    task1.calhist(window_start)
    task1.findmedian(window_start)
    for j in range(blur_sp.shape[1]-kernel_size[1]):
        if j == 0:
            median = task1.get_median()
            output[i][j] = median
            continue
        window = blur_sp[i:i+kernel_size[0], j:j+kernel_size[1]]
        task1.update(window)
        median = task1.get_median()
        output[i][j] = median

end = time.time()
print("time consumed：{:.4f}".format(end-start))
cv2.imshow("Lenna", image)
cv2.imshow("Spice salt noise image", blur_sp)
cv2.imshow("output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
