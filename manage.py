# -*- coding:utf-8 -*-
# Created on :2020/7/5 11:08
# Author:tzh

from main import model1, model2, find_position, split_char

colors, card_imgs = find_position("./img/京H99999.jpg")
res, colors, img, res_img = split_char(colors, card_imgs, model1, model2)


# from main1 import model1, model2, find_position, split_char

# colors, card_imgs = find_position("./img/黑AD03210.jpg")
# res, colors, img = split_char(colors, card_imgs, model1, model2)
