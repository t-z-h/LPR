#!/usr/bin/env python
# -*- coding: utf-8 -*-
# lyxiang


from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
import threading
from main import find_position, split_char, model1, model2


class Application(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master, bg='white')
        self.pack(expand=YES, fill=BOTH)
        self.window_init()
        self.createWidgets()

    # 基本框架
    def window_init(self):
        # title文字
        self.master.title('车牌识别系统')
        self.master.bg = 'black'
        # 窗口弹出大小
        width, height = 1900, 750
        self.master.geometry("{}x{}".format(width, height))

    def createWidgets(self):
        # fm1 头部标题
        # self.fm1 = Frame(self, bg='black')
        self.titleLabel = Label(self.window_init(), text="车牌识别系统", font=('微软雅黑', 32),
                                fg="white", bg='black').place(x=630, y=20)

        # fm2 车牌号botton 车牌识别botton

        self.predictButt = Button(self.window_init(), text='车牌号：', bg='black', fg='white',
                                  font=('微软雅黑', 20), width='10').place(x=800,
                                                                       y=580)  # , command=self.output_predict_sentence
        # print(self.predictButt)
        self.predictEntry = Entry(self.window_init(), font=('微软雅黑', 25),
                                  width='20', fg='black')
        self.predictEntry.place(x=970, y=580)
        # self.predictEntry.pack()
        self.truthButto = Button(self.window_init(), text='车牌类型：', bg='black', fg='white',
                                 font=('微软雅黑', 20), width='10').place(x=800, y=640)
        self.truthEntry = Entry(self.window_init(), font=('微软雅黑', 25),
                                width='20', fg='#22C9C9')
        self.truthEntry.place(x=970, y=640)

        # fm3 图片显示

        load_02 = Image.open('./GUI_img/002.png').resize((500, 200))
        initIamge_02 = ImageTk.PhotoImage(load_02)
        self.panel1 = Label(self.window_init(), image=initIamge_02)  # .place(x=200,y=200)
        self.panel1.image = initIamge_02
        self.panel1.place(x=890, y=250)

        load = Image.open('./GUI_img/0_爱奇艺.jpg').resize((500, 400))
        initIamge = ImageTk.PhotoImage(load)
        self.panel = Label(self.window_init(), image=initIamge)  # .place(x=200,y=200)
        self.panel.image = initIamge
        self.panel.place(x=150, y=150)

        # fm4
        self.predictButto = Button(self.window_init(), text='上传文件', bg='#22C9C9', fg='white',
                                   font=('微软雅黑', 20), width='10',
                                   command=self.up_file_path).place(x=130, y=640)

        self.truthButt = Button(self.window_init(), text='识别文件', bg='#FF4081', fg='white',
                                font=('微软雅黑', 20), width='10',
                                command=self.output_predict_sentence).place(x=500, y=640)

    # 文本框1输出内容
    def output_predict_sentence(self):
        color, card_imgs = find_position(file_path)
        res, color, img, res_img = split_char(color, card_imgs, model1, model2)
        if color == "yello":
            color = "黄牌"
            self.truthEntry.configure(fg='#FF8C00')
        if color == "blue":
            color = "蓝牌"
            self.truthEntry.configure(fg='blue')
        if color == "green":
            color = "绿牌"
            self.truthEntry.configure(fg='green')
        ground_truth = color
        predicted_sentence_str = res
        res = res.insert(2, "-")
        # self.predictEntry.delete(0, END)

        # print(self.predictEntry)
        predicted_sentence_str = "".join(predicted_sentence_str)
        # print(predicted_sentence_str)
        self.predictEntry.delete(0, END)
        self.predictEntry.insert(0, predicted_sentence_str)
        self.truthEntry.delete(0, END)
        self.truthEntry.insert(0, ground_truth)
        load = Image.open(res_img).resize((500, 100))
        img = ImageTk.PhotoImage(load)
        self.panel1.config(image=img)
        self.panel1.image = img

    # 上传图片函数
    def up_file_path(self):
        global file_path
        file_path = filedialog.askopenfilename()
        a = file_path
        print(a)
        # # 二进制形式读取文件
        # # with open(a, "rb")as f:
        # #     print(f.read())
        load = Image.open(a).resize((500, 400))
        print(load)
        initIamge = ImageTk.PhotoImage(load)
        self.panel.config(image=initIamge)
        self.panel.image = initIamge


if __name__ == '__main__':
    app = Application()
    # to do
    app.mainloop()
