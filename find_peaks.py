# -*- coding:utf-8 -*-
# Created on :2020/7/9 10:05
# Author:tzh
import cv2
import numpy as np


# 截取每个字符的位置
def find_waves(threshold, histogram):
    up_point = -1  # 上升点
    is_peak = False    # 默认不是最高点
    wave_peaks = []
    for i, x in enumerate(histogram):
        # 如果它是最高点，并且x在阈值范围内
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        # 如果它不是最高点，且x大于设定阈值，表示它为最高点
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    # print("wave_peaks", wave_peaks)
    return wave_peaks


# 查找水平直方图波峰
def find_level_weak(gray_img):
    # plt.imshow(gray_img)
    # plt.show()
    # (h, w) = gray_img.shape
    # a = [0 for z in range(0, w)]
    # print(a)  # a = [0,0,0,0,0,0,0,0,0,0,...,0,0]初始化一个长度为w的数组，用于记录每一列的黑点个数
    # gray_img_copy = gray_img.copy()
    # # 记录每一列的波峰
    # for j in range(0, w):  # 遍历一列
    #     for i in range(0, h):  # 遍历一行
    #         if gray_img_copy[i, j] == 0:  # 如果改点为黑点
    #             a[j] += 1  # 该列的计数器加一计数
    #             gray_img_copy[i, j] = 255  # 记录完后将其变为白色
    #     # print (j)
    #
    # #
    # for j in range(0, w):  # 遍历每一列
    #     for i in range((h - a[j]), h):  # 从该列应该变黑的最顶部的点开始向最底部涂黑
    #         gray_img_copy[i, j] = 0  # 涂黑
    #
    # # 此时的thresh1便是一张图像向垂直方向上投影的直方图
    # # 如果要分割字符的话，其实并不需要把这张图给画出来，只需要的到a=[]即可得到想要的信息
    #
    # plt.imshow(gray_img_copy, cmap=plt.gray())
    # plt.show()

    x_histogram = np.sum(gray_img, axis=1)
    # print("x_histogram", x_histogram)
    x_min = np.min(x_histogram)
    # 取阈值
    x_average = np.sum(x_histogram) / x_histogram.shape[0]
    x_threshold = (x_min + x_average) / 2
    # print("x_threshold", x_threshold)
    wave_peaks = find_waves(x_threshold, x_histogram)
    return wave_peaks


# 查找垂直直方图波峰
def find_vertical_weak(gray_img):
    # plt.imshow(gray_img)
    # plt.show()
    # (h, w) = gray_img.shape
    # a = [0 for z in range(0, w)]
    # print(a)  # a = [0,0,0,0,0,0,0,0,0,0,...,0,0]初始化一个长度为w的数组，用于记录每一列的黑点个数
    # gray_img_copy = gray_img.copy()
    # # 记录每一列的波峰
    # for j in range(0, w):  # 遍历一列
    #     for i in range(0, h):  # 遍历一行
    #         if gray_img_copy[i, j] == 0:  # 如果改点为黑点
    #             a[j] += 1  # 该列的计数器加一计数
    #             gray_img_copy[i, j] = 255  # 记录完后将其变为白色
    #     # print (j)
    #
    # #
    # for j in range(0, w):  # 遍历每一列
    #     for i in range((h - a[j]), h):  # 从该列应该变黑的最顶部的点开始向最底部涂黑
    #         gray_img_copy[i, j] = 0  # 涂黑
    #
    # # 此时的thresh1便是一张图像向垂直方向上投影的直方图
    # # 如果要分割字符的话，其实并不需要把这张图给画出来，只需要的到a=[]即可得到想要的信息
    #
    # plt.imshow(gray_img_copy, cmap=plt.gray())
    # plt.show()

    # 水平投影
    # img1 = gray_img.copy()
    # (h, w) = img1.shape
    #
    # # 初始化一个跟图像高一样长度的数组，用于记录每一行的黑点个数
    # a = [0 for z in range(0, h)]
    #
    # for i in range(0, h):  # 遍历每一行
    #     for j in range(0, w):  # 遍历每一列
    #         if img1[i, j] == 0:  # 判断该点是否为黑点，0代表黑点
    #             a[i] += 1  # 该行的计数器加一
    #             img1[i, j] = 255  # 将其改为白点，即等于255
    # for i in range(0, h):  # 遍历每一行
    #     for j in range(0, a[i]):  # 从该行应该变黑的最左边的点开始向最右边的点设置黑点
    #         img1[i, j] = 0  # 设置黑点
    # # cv2.imshow("img", img1)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
    # ret, img1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # plt.imshow(img1, "gray")
    # plt.show()

    row_num, col_num = gray_img.shape
    # 去掉车牌上下边缘1个像素，避免白边影响阈值判断
    gray_img = gray_img[1:row_num - 1]
    y_histogram = np.sum(gray_img, axis=0)
    y_min = np.min(y_histogram)
    y_average = np.sum(y_histogram) / y_histogram.shape[0]
    y_threshold = (y_min + y_average) / 5  # U和0要求阈值偏小，否则U和0会被分成两半

    wave_peaks = find_waves(y_threshold, y_histogram)
    return wave_peaks

# 根据找出的波峰，分隔图片，从而得到逐个字符图片
import matplotlib.pyplot as plt
def seperate_card(img, waves):
    part_cards = []
    for wave in waves:
        # 截取每个字符
        part_cards.append(img[:, wave[0]:wave[1]])
        plt.imshow(img[:, wave[0]:wave[1]], "gray")
        plt.show()
    return part_cards


# gray_img = cv2.imread("./result/1.png", 0)
# wave_peaks = find_level_weak(gray_img)
# wave = max(wave_peaks, key=lambda x: x[1] - x[0])
# gray_img = gray_img[wave[0]:wave[1]+1]
# find_vertical_weak(gray_img)
# wave = max(wave_peaks, key=lambda x: x[1] - x[0])
# part_cards = seperate_card(gray_img, wave_peaks)


