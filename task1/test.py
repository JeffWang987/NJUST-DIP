# -*- coding: utf-8 -*-
# @Time    : 2019/11/30 15:28
# @Author  : Jeff Wang
# @Email   : jeffwang987@163.com   OR    wangxiaofeng2020@ia.ac.cn
# @Software: PyCharm
"""
本次作业：实现二维的快速中值滤波。
论文参考：A Fast Two-Dimensional Median Filtering Algorithm
"""
import cv2
import numpy as np
from task1 import imutils
import time

"""给原图添加噪声"""
image = cv2.imread("Lenna.jpg")
utils = imutils.Utils()
gray = utils.rgb2gray(image)
blur_sp = utils.sp_noise(gray, prob=0.01)
# cv2.imshow("Lenna", image)
# cv2.imshow("Spice salt noise image", blur_sp)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

start = time.time()
"""执行中值滤波算法"""
output = np.copy(blur_sp)
output = cv2.medianBlur(image, 5)
end = time.time()
print("time consumed：{:.4f}".format(end-start))
cv2.imshow("output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

