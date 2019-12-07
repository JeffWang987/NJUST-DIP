# -*- coding: utf-8 -*-
# @Time    : 2019/12/7 14:49
# @Author  : Jeff Wang
# @Email   : jeffwang987@163.com   OR    wangxiaofeng2020@ia.ac.cn
# @Software: PyCharm
import cv2
import numpy as np
import matplotlib.pyplot as plt


def fft_distance(m, n):
    """
    功能：定义一个函数确定mxn的矩阵中的每个元素距离中心的距离(给巴特沃斯和高斯滤波器用)
    :param m:高
    :param n: 宽
    :return: 距离矩阵
    """
    u = np.array([np.abs(i-m/2) for i in range(m)])
    v = np.array([np.abs(i-n/2) for i in range(n)])
    u.shape = m, 1
    v.shape = 1, n  # 两个都是一维矩阵，但是方向不同，相加会broadcast
    dist = np.sqrt(u**2 + v**2)
    return dist


def gs_filter(fshift, scale):
    d = fft_distance(fshift.shape[0], fshift.shape[1])
    mask_gs = np.exp(- (d**2) / (2*scale**2))
    mask_gs = 1 - mask_gs
    return mask_gs


def bt_filter(fshift, d0=20, n=2):
    d = fft_distance(fshift.shape[0], fshift.shape[1])
    mask_butterworth = 1/(1+(d/d0)**(2*n))
    mask_butterworth = 1 - mask_butterworth
    return mask_butterworth


def Homomorphic_filter(img, scale=30):
    """
    同态增清晰，暗部的细节体现出来。
    :param img:
    :param scale:
    :return:
    """
    # img = np.log(img + 0.1)
    img_fft = np.fft.fft2(img)
    img_fft_shift = np.fft.fftshift(img_fft)
    # mask = gs_filter(img_fft_shift, scale)
    mask = bt_filter(img_fft_shift, d0=scale, n=2)
    img_done = mask*img_fft_shift
    img_ishift = np.fft.ifftshift(img_done)
    img_ifft = np.fft.ifft2(img_ishift)
    img_back = np.abs(img_ifft)
    # img_back = np.exp(img_back)
    return img_back


if __name__ == "__main__":
    img1 = cv2.imread("1.jpg", 0)
    # img1 = img1[60:170, :200]
    out1 = Homomorphic_filter(img1, 0.5)
    plt.subplot(1, 2, 1), plt.imshow(img1, 'gray'), plt.title('img1')
    plt.subplot(1, 2, 2), plt.imshow(out1, 'gray'), plt.title('img_back1')
    plt.show()
