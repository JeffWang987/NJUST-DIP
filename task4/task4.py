# -*- coding: utf-8 -*-
# @Time    : 2019/12/22 10:12
# @Author  : Jeff Wang
# @Email   : jeffwang987@163.com   OR    wangxiaofeng2020@ia.ac.cn
# @Software: PyCharm
"""
本次作业：在RGB和HSV空间中实现图像分割。
使用算法：Kmeans
作业感悟：emmm，有两个问题，第一个是：RGB和HSV的效果差不多
                            第二个是：若分类时有一个类别中没有像素归属时，会出现除0错误，有时间处理一下吧。
"""
import numpy as np
import cv2


def RandCentriod_RGB(K):
    centroid_r = np.random.randint(0, 255, [K, 1])
    centroid_g = np.random.randint(0, 255, [K, 1])
    centroid_b = np.random.randint(0, 255, [K, 1])
    centroid = []
    for i in range(K):
        centroid.append(np.concatenate((centroid_b, centroid_g, centroid_r), axis=1)[i])
    return centroid


def RandCentriod_HSV(K):
    centroid_h = np.random.randint(0, 255, [K, 1])
    centroid_s = np.random.randint(0, 255, [K, 1])
    centroid_v = np.random.randint(0, 255, [K, 1])
    centroid = []
    for i in range(K):
        centroid.append(np.concatenate((centroid_h, centroid_s, centroid_v), axis=1)[i])
    return centroid


def draw(img, length, index, K, flag):
    # 5. 计算掩膜，准备画图
    mask = []
    for i in range(K):
        mask.append(np.zeros(img.shape, dtype=np.uint8))
    # 6. 画图
    out = []
    for i in range(K):
        if length[i] > 1:
            for j in range(length[i]):
                mask[i][np.squeeze(index[i])[j] // img.shape[1], np.squeeze(index[i])[j] % img.shape[1]] = 1
            out.append(mask[i] * img)
            if flag == 'HSV':
                out[i] = cv2.cvtColor(out[i], cv2. COLOR_HSV2BGR)
            cv2.imshow('{}'.format(i), out[i])
    if flag == 'HSV':
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    cv2.imshow('ori', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # for i in range(K):
    #     if flag == 'HSV':
    #         cv2.imwrite('HSVout{}.jpg'.format(i), out[i])
    #     else:
    #         cv2.imwrite('RGBout{}.jpg'.format(i), out[i])


def KMeans_RGB(img, K, flag='HSV'):
    if flag == 'RGB':
        centroid = RandCentriod_RGB(K)
    else:
        centroid = RandCentriod_HSV(K)
    # 1. 把K个中心复制img.shape[0]*img.shape[1]遍，也就是展开，为了之后矩阵操作更快
    centroid_reshape = []
    for i in range(K):
        centroid_reshape.append(np.tile(centroid[i], (img.shape[0] * img.shape[1], 1)))
    # 2. 把原图像也展开，而不是二维排列，为了和上面的质心矩阵一样size，矩阵操作更快
    img_reshape = img.reshape(img.shape[0] * img.shape[1], 3)
    # 3. dist用来存储每个像素RGB三个值和质心RGB的欧氏距离
    dist = np.zeros([img.shape[0] * img.shape[1], K], dtype=np.float32)
    # 4. 设定好初始的J(确保在第一轮迭代后，J_last=9999不会小于真正计算出来的J)，开始迭代
    J_last = 99999
    J = 9999
    t = 100  # 运行次数，一般10次以内，不然直接退出
    while J < J_last and t > 0:
        print(t)
        t = t - 1
        J_last = J
        # 0.分别计算三个到质心的距离
        for i in range(K):
            if flag == 'RGB':
                dist[:, i] = np.linalg.norm(centroid_reshape[i] - img_reshape, axis=1)
            else:
                dist[:, i] = np.linalg.norm((centroid_reshape[i] - img_reshape)[:, :-1], axis=1)
        # 1. 取最小距离的那个质心作为该像素的标号
        label = np.argmin(dist, axis=1)
        index = []
        for i in range(K):
            index.append(np.argwhere(label == i))
        # 2. 看看该质心下跟了多少个像素
        length = []
        for i in range(K):
            length.append(len(index[i]))
        # 3. 根据这些像素重新分配质心
        for i in range(K):
            centroid[i] = np.round(1 / length[i] * np.sum(img_reshape[np.squeeze(index[i], axis=1)], axis=0))
        # 4. 把质心展开，方便矩阵操作
        for i in range(K):
            centroid_reshape[i] = np.tile(centroid[i], (img.shape[0] * img.shape[1], 1))
        # 计算代价函数，在我们这里是所有像素到其质心的距离之和
        J = []
        for i in range(K):
            J.append(np.sum(np.linalg.norm(img_reshape[np.squeeze(index[i], axis=1)] - np.tile(centroid[i], (length[i], 1)), axis=1)))
        J = np.sum(J) / (img.shape[0] * img.shape[1])
    print(length)
    draw(img, length, index, K, flag)


if __name__ == "__main__":
    K = 10
    img = cv2.imread('b4.jpg')
    # KMeans_RGB(img, K, 'RGB')
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    KMeans_RGB(img_hsv, K)








