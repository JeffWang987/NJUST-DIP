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
import imutils
import numpy as np
import time

"""给原图添加噪声"""
image = cv2.imread("Lenna.jpg")  # 读取图像
utils = imutils.Utils()  # 对象实例化，这个用于RGB转灰度、增加噪声等
gray = utils.rgb2gray(image)  # 转为灰度图像
blur_sp = utils.sp_noise(gray, prob=0.01)  # 添加1%的椒盐噪声

"""执行快速中值滤波算法"""
start = time.time()  # 算法开始计时
output = np.copy(blur_sp)  # 先为输出图像创建空间
kernel_size = (3, 3)  # 我们的中值滤波模板
task1 = imutils.FastMedianBlur(blur_sp, kernel_size)  # 对象实例化，这个用于快速中值滤波算法
for i in range(blur_sp.shape[0]-kernel_size[0]):  # 行循环
    # 每次到一个新行的时候，要把直方图清零
    task1.hist2zero()
    # 每一行第一个window的截取
    window_start = blur_sp[i:i+kernel_size[0], 0:kernel_size[1]]
    # 计算每一行第一个window的直方图
    task1.calhist(window_start)
    # 利用希尔排序求每一行第一个window的中值
    task1.findmedian(window_start)
    for j in range(blur_sp.shape[1]-kernel_size[1]):  # 列循环
        # 因为每一行第一个求过了，可以跳过
        if j == 0:
            median = task1.get_median()
            output[i][j] = median
            continue
        # 截取window
        window = blur_sp[i:i+kernel_size[0], j:j+kernel_size[1]]
        # 按照论文方法更新直方图
        task1.update(window)
        # 按照论文方法更新中值
        median = task1.get_median()
        # 把中值代替原来的像素灰度
        output[i][j] = median

end = time.time()  # 算法开始计时
print("time consumed：{:.4f}".format(end-start))  # 显示算法用时
cv2.imshow("Original", image)  # 显示对比原图、噪声图、滤波图
cv2.imshow("Spice salt noise image", blur_sp)
cv2.imshow("output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
