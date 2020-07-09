# -*- coding:utf-8 -*-
# Created on :2020/7/9 10:05
# Author:tzh
import numpy as np


def find_waves(threshold, histogram):
    # print(len(histogram))
    # plt.plot(histogram)
    # plt.show()
    up_point = -1  # 上升点
    is_peak = False
    # print("threshold", threshold)
    # print("histogram", histogram)
    # if histogram[0] > threshold:
    #     up_point = 0
    #     is_peak = True
    wave_peaks = []
    for i, x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    # print("wave_peaks", wave_peaks)
    return wave_peaks


# 查找水平直方图波峰
def find_level_weak(gray_img):
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
        # print(gray_img.shape)
        row_num, col_num = gray_img.shape
        # 去掉车牌上下边缘1个像素，避免白边影响阈值判断
        gray_img = gray_img[1:row_num - 1]
        y_histogram = np.sum(gray_img, axis=0)
        y_min = np.min(y_histogram)
        y_average = np.sum(y_histogram) / y_histogram.shape[0]
        y_threshold = (y_min + y_average) / 5  # U和0要求阈值偏小，否则U和0会被分成两半

        wave_peaks = find_waves(y_threshold, y_histogram)
        return wave_peaks

