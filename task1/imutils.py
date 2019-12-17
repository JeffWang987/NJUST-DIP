# -*- coding: utf-8 -*-
# @Time    : 2019/11/30 14:59
# @Author  : Jeff Wang
# @Email   : jeffwang987@163.com   OR    wangxiaofeng2020@ia.ac.cn
# @Software: PyCharm

import numpy as np
import random


class FastMedianBlur:
    """定义一个类供TASK1使用"""
    def __init__(self, image, kernel_size):
        self.window_width = kernel_size[1]
        self.window_height = kernel_size[0]
        self.median = 0
        self.th = self.window_height*self.window_width//2+1
        self.hist = np.zeros((image.shape[0]*image.shape[1], 1), dtype="uint8")
        self.left = np.zeros((self.window_height, 1), dtype="uint8")
        self.right = np.zeros((self.window_height, 1), dtype="uint8")

    def hist2zero(self):
        """
        功能：每到新的一行需要把hist清零。
        """
        self.hist = 0 * self.hist

    def calhist(self, window):
        """
        功能：计算直方图
        :param window:被设置的kernel所框住image的部分
        """
        self.left = window[:, 0]
        for i in range(self.window_height):
            for j in range(self.window_width):
                grayvalue = window[i][j]
                self.hist[grayvalue] += 1

    def findmedian(self, window):
        """
        功能：利用希尔排序实现寻找中值
        :param window: 被设置的kernel所框住image的部分
        """
        window = np.reshape(window, (1, -1))
        inc = window.shape[1]//3 + 1
        while inc > 1:
            for i in range(window.shape[1]-inc):
                if window[0][i] > window[0][i+inc]:
                    temp = window[0][i]
                    window[0][i] = window[0][i+inc]
                    window[0][i+inc] = temp
            inc = inc//3 + 1
            if inc == 1:
                break
        self.median = window[0][window.shape[1]//2+1]

    def update_hist(self):
        """
        功能：当窗口移动后，减去左边一列，加上右边一列，更新hist
        """
        for i in range(self.window_height):
            left = self.left[i]
            right = self.right[i]
            self.hist[left] -= 1
            self.hist[right] += 1

    def update_median(self):
        """
        功能：更新中值
        """
        if np.sum(self.hist[:self.median]) == self.th:
            self.median = self.median
        elif np.sum(self.hist[:self.median]) < self.th:
            while np.sum(self.hist[:self.median]) < self.th:  # 之前的错误：把return放在了while里面，while只执行了一遍就return了
                self.median += 1
        else:
            while np.sum(self.hist[:self.median]) > self.th:
                self.median -= 1
            self.median += 1

    def update(self, window):
        """
        功能：在这个函数里left、right、hist、median都会被更新
        :param window: 同上
        """
        self.right = window[:, -1]
        self.update_hist()
        self.update_median()
        self.left = window[:, 0]

    def get_median(self):
        """
        :return: 返回self.median
        """
        return self.median


class Utils:

    def __init__(self):
        pass

    def rgb2gray(self, image):
        """
        将RGB图像转到灰度空间
        :param image: 原图像
        :return gray: 灰度图像
        """
        gray = np.zeros(image.shape[:2], dtype="uint8")
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                gray[i][j] = np.sum(image[i][j])/3
        return gray

    def sp_noise(self, image, prob):
        """
        椒盐英文：spiced salt
        功能：添加椒盐噪声
        :param image: 原图片。
        :param prob: 噪声比例,若产生的随机数小于prob/2，则该点变为黑色，若大于1-prob/2，则产生白色。只有在(prob/2, 1-prob/2)这个范围内灰度不变。故prob的 值应该小于1大于0
        :return output: 增加噪声后的图片
        """
        if prob < 0:
            print("prob的值应该在0至0.5之间,已自动设置为0！")
            prob = 0
        elif prob > 1:
            print("prob的值应该在0至0.5之间,已自动设置为1！")
            prob = 1
        output = np.zeros(image.shape, dtype="uint8")
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob/2:
                    output[i][j] = 0
                elif rdn > (1-prob/2):
                    output[i][j] = 255
                else:
                    output[i][j] = image[i][j]
        return output

    def gasuss_noise(self, image, mean=0, std=5):
        """
        功能：添加高斯噪声
        :param mean: 高斯核均值
        :param std: 高斯核标准差
        :return output: 增加噪声后的图片
        """
        output = np.uint8(np.copy(image))
        h, w, c = output.shape
        for row in range(0, h):
            for col in range(0, w):
                for chan in range(0, c):
                    s = np.random.normal(mean, std, 1)  # 产生随机数
                    output[row, col, chan] += np.int8(s)
                    if output[row, col, chan] > 255:
                        output[row, col, chan] = 255
                    elif output[row, col, chan] < 0:
                        output[row, col, chan] = 0
        return np.uint8(output)
