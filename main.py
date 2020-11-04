# -*- coding:utf-8 -*-
# Created on :2020/6/22 10:53
# Author:tzh
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
from find_peaks import find_level_weak, find_vertical_weak

SZ = 20  # 训练图片长宽
MAX_WIDTH = 1000  # 原始图片最大宽度
Min_Area = 2000  # 车牌区域允许最大面积
PROVINCE_START = 1000

# 配置参数
config = {
    "open": 1,
    "blur": 3,
    "morphologyr": 4,
    "morphologyc": 19,
    "col_num_limit": 10,
    "row_num_limit": 21
}


def point_limit(point):
    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0


def accurate_place(card_img_hsv, limit_min, limit_max, color):
    """
    定位车牌精确位置，并返回
    """
    # show_img(card_img_hsv)
    row_num, col_num = card_img_hsv.shape[:2]
    # print("col_num, row_num", col_num, row_num)
    # xl = 图片的column长度，yl = 图片的row长度
    xl = col_num
    xr = 0
    yh = 0
    yl = row_num
    row_num_limit = config["row_num_limit"]
    col_num_limit = col_num * 0.8
    # col_num_limit = col_num * 0.8 if color != "green" else col_num * 0.6  # 绿色色车牌白色区域不需要
    # print(col_num_limit)
    # 找每个点的H, S, V的值，如果在指定范围内count+1，如果count>col_num_limit，再进行判断
    for i in range(row_num):    # row_num=轮廓行的长度
        count = 0
        for j in range(col_num):
            H = card_img_hsv.item(i, j, 0)
            S = card_img_hsv.item(i, j, 1)
            V = card_img_hsv.item(i, j, 2)
            # print("H S V", H, S, V)
            if limit_min < H <= limit_max and 34 < S and 46 < V:
                count += 1
        if count > col_num_limit:
            # print("yl, i", yl, i)
            # 找到截取后列的起始位置yl
            if yl > i:
                yl = i
            # print("yh, i", yh, i)
            # 找到截取后列的终止位置yh
            if yh < i:
                yh = i
            # print("count", count)
    for j in range(col_num):
        count = 0
        for i in range(row_num):
            H = card_img_hsv.item(i, j, 0)
            S = card_img_hsv.item(i, j, 1)
            V = card_img_hsv.item(i, j, 2)
            if limit_min < H <= limit_max and 34 < S and 46 < V:
                count += 1
        if count > row_num - row_num_limit:
            # print("xl, j", xl, j)
            # 找到截取后行的起始位置xl
            if xl > j:
                xl = j
            # 找到截取后行的终止位置xr
            if xr < j:
                xr = j
    return xl, xr, yh, yl


# 根据找出的波峰，分隔图片，从而得到逐个字符图片
def seperate_card(img, waves):
    part_cards = []
    for wave in waves:
        # 截取每个字符
        part_cards.append(img[:, wave[0]:wave[1]])
        # plt.imshow(img[:, wave[0]:wave[1]], "gray")
        # plt.show()
    return part_cards


def show_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()


def find_position(car_path):
    print(config)
    if type(car_path) == type(""):
        img = cv2.imdecode(np.fromfile(car_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        # show_img(img)
    else:
        img = car_path
    # print(img.shape[:2])
    pic_hight, pic_width = img.shape[:2]
    # print(MAX_WIDTH, pic_width)
    # 假如图片过大，需要指定大小
    if pic_width > MAX_WIDTH:
        # print(MAX_WIDTH / pic_width)
        resize_rate = MAX_WIDTH / pic_width
        img = cv2.resize(img, (MAX_WIDTH, int(pic_hight * resize_rate)), interpolation=cv2.INTER_AREA)

    blur = config["blur"]
    img_copy = img
    show_img(img_copy)
    # 高斯去噪
    if blur > 0:
        img = cv2.GaussianBlur(img, (blur, blur), 0)  # 图片分辨率调整
    # show_img(img)
    oldimg = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ##### 去掉图像中不会是车牌的区域 ##### #
    kernel = np.ones((20, 20), np.uint8)
    # 形态学操作
    img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # show_img(img_opening)
    img_opening = cv2.addWeighted(img, 1, img_opening, -1, 0)
    show_img(img_opening)
    # 找到图像边缘
    ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    show_img(img_opening)
    img_edge = cv2.Canny(img_thresh, 100, 200)
    show_img(img_edge)
    # sobel_img = cv2.Sobel(img_opening, -1, 1, 0, ksize=3)
    # ret, img_edge = cv2.threshold(sobel_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # show_img(img_edge)

    # 使用开运算和闭运算让图像边缘成为一个整体
    kernel = np.ones((config["morphologyr"], config["morphologyc"]), np.uint8)
    # 闭运算：表示先进行膨胀操作，再进行腐蚀操作
    img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
    # 开运算：表示的是先进行腐蚀，再进行膨胀操作
    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)
    show_img(img_edge1)
    show_img(img_edge2)

    # 查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
    contours, hierarchy = cv2.findContours(
        img_edge2,
        cv2.RETR_TREE,   # cv2.RETR_TREE建立一个等级树结构的轮廓
        cv2.CHAIN_APPROX_SIMPLE   # cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
    )
    # 筛选车牌轮廓，根据面积筛选
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > Min_Area]
    print('轮廓', len(contours))
    # 一一排除不是车牌的矩形区域
    car_contours = []
    box_ = []
    for cnt in contours:
        # 获取最小外接矩形，返回值为：[中心点坐标，(宽度，高度)，旋转角度]
        rect = cv2.minAreaRect(cnt)
        # print(rect)
        # 最小外接矩形的中心（width，height）
        area_width, area_height = rect[1]
        if area_width < area_height:
            area_width, area_height = area_height, area_width
        # 计算长宽比，便于下面判断
        wh_ratio = area_width / area_height
        # 要求矩形区域长宽比在2到5.5之间，2到5.5是车牌的长宽比，其余的矩形排除
        if wh_ratio > 2 and wh_ratio < 5.5:
            # print(rect)
            car_contours.append(rect)
            # 以下代码可以不要，只是用用来查看筛选出来的轮廓用到
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            box_.append(box)
    # 画框
    # for box in box_:
    #     oldimg = cv2.drawContours(img_copy, [box], 0, (0, 0, 255), 2)
    # cv2.imwrite("./q.jpg", oldimg)
    # show_img(oldimg)

    # 查看轮廓

    # print(car_contours)
    # for i in range(len(contours)):
    #     x, y, w, h = cv2.boundingRect(contours[i])
    #     newimage = img_copy[y:y + h+10, x:x + w]
    #     show_img(newimage)

    # print(box_)
    # box_ = box_[0]
    # xl, xr, yh, yl = box_[0][0], box_[2][0], box_[2][1], box_[0][1]
    # print(xl, xr, yh, yl)
    # p = img_copy[xl:xr, yl:yh]
    # print(p)
    # show_img(p)
    # print(car_contours)


    card_imgs = []
    # 矩形区域可能是倾斜的矩形，需要矫正，以便使用颜色定位
    for rect in car_contours:
        # print(rect)
        if rect[2] > -1 and rect[2] < 1:  # 创造角度，使得左、高、右、低拿到正确的值
            angle = 1
        else:
            angle = rect[2]
        rect = (rect[0], (rect[1][0] + 3, rect[1][1] + 5), angle)  # 扩大范围，避免车牌边缘被排除
        # 获取四个顶点
        box = cv2.boxPoints(rect)
        heigth_point = right_point = [0, 0]
        left_point = low_point = [pic_width, pic_hight]
        # 得到四个顶点, 根据外接矩形的四个顶点来确定
        for point in box:
            if left_point[0] > point[0]:
                left_point = point
            if low_point[1] > point[1]:
                low_point = point
            if heigth_point[1] < point[1]:
                heigth_point = point
            if right_point[0] < point[0]:
                right_point = point
        print(left_point, low_point, heigth_point, right_point)
        if left_point[1] <= right_point[1]:  # 正角度
            # 取到变换后右下角的点的位置
            new_right_point = [right_point[0], heigth_point[1]]
            # 三个点的原坐标
            pts1 = np.float32([left_point, heigth_point, right_point])
            # 变换后三个点的坐标
            pts2 = np.float32([left_point, heigth_point, new_right_point])  # 字符只是高度需要改变
            # 得到变换矩阵
            M = cv2.getAffineTransform(pts1, pts2)
            # 仿射变换
            dst = cv2.warpAffine(img_copy, M, (pic_width, pic_hight))
            # 负数处理
            point_limit(new_right_point)
            point_limit(heigth_point)
            point_limit(left_point)
            # 截取角度矫正后的图像
            card_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]
            card_imgs.append(card_img)

        elif left_point[1] > right_point[1]:  # 负角度
            new_left_point = [left_point[0], heigth_point[1]]
            pts2 = np.float32([new_left_point, heigth_point, right_point])  # 字符只是高度需要改变
            pts1 = np.float32([left_point, heigth_point, right_point])
            M = cv2.getAffineTransform(pts1, pts2)
            dst = cv2.warpAffine(img_copy, M, (pic_width, pic_hight))
            point_limit(right_point)
            point_limit(heigth_point)
            point_limit(new_left_point)
            card_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]
            card_imgs.append(card_img)
    # 查看矫正后的图像
    # for car_img in card_imgs:
    #     show_img(car_img)
    # show_img(img_copy)
    # print(card_imgs)
    colors = []
    # ####确定车牌颜色#### #
    for card_index, card_img in enumerate(card_imgs):
        green = yellow = blue = 0
        # show_img(card_img)
        # 转换HSV
        card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
        # 有转换失败的可能，原因来自于上面矫正矩形出错
        if card_img_hsv is None:
            continue
        # 获取行的长度和列的长度
        row_num, col_num = card_img_hsv.shape[:2]
        # 得到像素点的总个数
        card_img_count = row_num * col_num
        # 渠道每一个像素点的各个通道的值
        for i in range(row_num):
            for j in range(col_num):
                H = card_img_hsv.item(i, j, 0)
                S = card_img_hsv.item(i, j, 1)
                V = card_img_hsv.item(i, j, 2)
                # print(H, S, V)
                # 像素点的值在指定的颜色范围内，该颜色的个数+1
                if 11 < H <= 34 and S > 34:
                    yellow += 1
                elif 35 < H <= 99 and S > 34:
                    green += 1
                elif 99 < H <= 124 and S > 34:
                    blue += 1
        # 定义color变量，默认没有颜色
        color = "no"
        # 定义颜色的取值上线和下限
        limit_min = limit_max = 0
        # 假如有一半的像素点为该颜色，则判定它为该颜色
        if yellow * 2 >= card_img_count:
            color = "yellow"
            limit_min = 11
            limit_max = 34  # 有的图片有色偏偏绿
        elif green * 2 >= card_img_count:
            color = "green"
            limit_min = 35
            limit_max = 99
        elif blue * 1.8 >= card_img_count:
            color = "blue"
            limit_min = 100
            limit_max = 124  # 有的图片有色偏偏紫
        colors.append(color)
        # print(blue, green, yellow, black, white, card_img_count)

        # cv2.imshow("color", card_img)
        # cv2.waitKey(0)

        if limit_min == 0:
            continue
        # 以上为确定车牌颜色
        # 以下为根据车牌颜色再定位，缩小边缘非车牌边界
        """
        yl为列的起始位置
        yh为列的终止位置
        xl为行的起始位置
        xr为行的终止位置
        """
        xl, xr, yh, yl = accurate_place(card_img_hsv, limit_min, limit_max, color)
        # show_img(card_img[yl:yh, xl + 2:xr])
        print(xl, xr, yh, yl)
        # 如果它们相等，则表示定位的车牌位置为空
        if yl == yh and xl == xr:
            continue
        need_accurate = False
        # 如果列的起始位置大于等于列的终止位置，则需要重新确认位置
        if yl >= yh:
            yl = 0
            yh = row_num
            need_accurate = True
        if xl >= xr:
            xl = 0
            xr = col_num
            need_accurate = True

        # show_img(card_img)

        # 截取车牌精确位置
        # 因为在上面进行了以颜色对车牌的精确定位，会导致绿色车牌上半部分的颜色渐变区域没有取到，所以在此要对绿牌做处理
        card_imgs[card_index] = card_img[yl:yh, xl + 2:xr] if color != "green" or yl < (yh - yl) // 4 else \
            card_img[yl - (yh - yl) // 4:yh, xl + 2:xr]
        # card_imgs[card_index] = card_img[yl:yh, xl + 2:xr]

        if need_accurate:  # 可能x或y方向未缩小，需要再试一次
            card_img = card_imgs[card_index]
            card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
            xl, xr, yh, yl = accurate_place(card_img_hsv, limit_min, limit_max, color)
            if yl == yh and xl == xr:
                continue
            if yl >= yh:
                yl = 0
                yh = row_num
            if xl >= xr:
                xl = 0
                xr = col_num
        card_imgs[card_index] = card_img[yl:yh, xl + 2:xr] if color != "green" or yl < (yh - yl) // 4 \
            else card_img[yl - (yh - yl) // 4:yh, xl + 2:xr]

    #     print(colors)
    # print(len(card_imgs))
    # for car_img in card_imgs:
    print(colors)
    # show_img(card_imgs[0])
    # 返回车牌颜色和精确定位的车牌图像
    return colors, card_imgs


def split_char(colors, card_imgs, model1, model2):
    for i, color in enumerate(colors):
        if color in ("blue", "yellow", "green"):
            card_img = card_imgs[i]
            # card_img = cv2.GaussianBlur(card_img, (3, 3), 0)
            # 去除车牌中的一些噪声点
            card_img = cv2.bilateralFilter(card_img, 9, 20, 20)
            show_img(card_img)
            # print("s", card_img.shape)
            gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
            # 黄、绿车牌字符比背景暗、与蓝车牌刚好相反，所以黄、绿车牌需要反向
            if color == "green" or color == "yellow":
                gray_img = cv2.bitwise_not(gray_img)

            # show_img(gray_img)

            # 二值化
            ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # show_img(gray_img)
            # 查找水平直方图波峰
            wave_peaks = find_level_weak(gray_img)

            # print("wave_peaks", wave_peaks[0])
            if len(wave_peaks) == 0:
                print("无波峰")
                continue
            # 水平方向的最大的波峰为车牌区域
            print(wave_peaks)
            wave = max(wave_peaks, key=lambda x: x[1] - x[0])
            print(wave)
            # show_img(gray_img)
            gray_img = gray_img[wave[0]:wave[1]+1]
            # show_img(gray_img)
            # 查找垂直直方图波峰
            wave_peaks = find_vertical_weak(gray_img)

            # for wave in wave_peaks:
            #     cv2.line(card_img, pt1=(wave[0], 5), pt2=(wave[1], 5), color=(0, 0, 255), thickness=2)
            # cv2.imshow("card", card_img)
            # cv2.waitKey(0)

            # 车牌字符数应大于6
            if len(wave_peaks) < 7:
                print("字符数量不够:", len(wave_peaks))
                continue
            # 拿到垂直方向宽度最大的字符区间
            wave = max(wave_peaks, key=lambda x: x[1] - x[0])
            # 取到宽度最大的字符区间的值
            max_wave_val = wave[1] - wave[0]
            # print(max_wave_dis / 3)
            # 判断是否是左侧车牌边缘
            # 当垂直方向第一个波峰位置长度 < max_wave_dis / 3 并且波峰的起始位置为0的时候，就把它列为车牌边缘，并删除它
            if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_val / 3 and wave_peaks[0][0] == 0:
                wave_peaks.pop(0)

            # 组合分离汉字，例如：川
            count = 0
            th_val = max_wave_val * 0.6
            for index, wave in enumerate(wave_peaks):
                now_wave = wave[1] - wave[0]
                if now_wave + count > th_val:
                    break
                else:
                    count += wave[1] - wave[0]
            if index > 0:
                wave = (wave_peaks[0][0], wave_peaks[index][1])
                wave_peaks = wave_peaks[index + 1:]
                wave_peaks.insert(0, wave)

            # 去除车牌上的分隔点
            point = wave_peaks[2]  # 取到车牌上的分割点
            if point[1] - point[0] < max_wave_val / 3:  # 如果point的波峰长度小于最长波峰的1/3
                point_img = gray_img[:, point[0]:point[1]]  # 获取到分割点在车牌上的位置
                if np.mean(point_img) < 255 / 5:  # 如果point_img均值低于255 / 5，则说明它的占面积很小，相当于一个点
                    wave_peaks.pop(2)  # 删除分割点

            if len(wave_peaks) < 7:  # 如果所有字符长度小于7，说明他不是车牌(车牌的长度应该大于6)
                print("字符数量不够:", len(wave_peaks))
                continue
            part_cards = seperate_card(gray_img, wave_peaks)
            # 显示每个字符
            # for part_card in part_cards:
            #     plt.imshow(part_card, "gray")
            #     plt.show()
            res = []
            for i, part_card in enumerate(part_cards):
                # 可能是固定车牌的铆钉
                if np.mean(part_card) < 255 / 5:
                    print("一个铆钉")
                    continue

                # cv2.imwrite(f"./vaild_img/c{i}.jpg", part_card)
                part_card_old = part_card

                # show_img(part_card)

                # 得到左右两边填充值
                w = abs(part_card.shape[1] - 20) // 2
                # 用0填充边框
                part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=0)
                # plt.imshow(part_card, "gray")
                # plt.show()

                # img = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_CUBIC)
                img = cv2.resize(part_card, (20, 20), interpolation=cv2.INTER_CUBIC)
                # img = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_CUBIC)


                th, img = cv2.threshold(img, 127, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

                # cv2.imwrite(f"vaild_img/c{i}.jpg", img)
                # plt.imshow(img, "gray")
                # plt.show()

                img = img.reshape((-1, SZ, SZ, 1))
                # cv2.imshow("img", img[0])
                # cv2.waitKey(0)
                img = tf.cast(img, tf.float32)

                if i == 0:
                    resp_ = model2.predict_classes(img)
                    resp = chinese_char[resp_[0]]
                else:
                    resp_ = model1.predict_classes(img)
                    resp = word_num[resp_[0]]
                # 判断最后一个数是否是车牌边缘，认假设车牌边缘被为是1
                if resp == "1" and i == len(part_cards) - 1:
                    if part_card_old.shape[0] / part_card_old.shape[1] >= 7:  # 1太细，认为是边缘
                        continue
                card_color = color
                res.append(resp)
        try:
            res_img = "./result/res.jpg"
            cv2.imwrite(res_img, card_img)
            print("车牌号为：", "".join(res))
            print("车牌颜色为", card_color)
            show_img(card_img)
            return res, card_color, card_img, res_img
        except:
            print("no")


# dir_ = os.listdir("./img")
# for i in dir_:
#     print(i)
#     color, img = predict(f"./img/{i}")
#     show_img(img)
# split_char(color, img)

chinese_char = ("川", "鄂", "赣", "甘", "贵", "桂", "黑", "沪", "冀", "津", "京", "吉", "辽", "鲁",
                "蒙", "闽", "宁", "靑", "琼", "陕", "苏", "晋", "皖", "湘", "新", "豫", "渝", "粤",
                "云", "藏", "浙")

word_num = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C",
            "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R",
            "S", "T", "U", "V", "W", "X", "Y", "Z")
model1 = keras.models.load_model('model/plate_model1_before3.h5')
model2 = keras.models.load_model('model/plate_model2-2.h5')
# colors, card_imgs = predict("./img/皖AUB816.jpg")
# res, colors, img = split_char(colors, card_imgs, model1, model2)
