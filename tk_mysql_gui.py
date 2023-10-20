"""

NAME : gui_tk_pro_max

USER : admin

DATE : 10/10/2023

PROJECT_NAME : RetinaFace-FaceNet

CSDN : friklogff
"""

import sys
from tkinter import *
from tkinter import ttk
import cv2
import time
import pymysql
from PIL import ImageTk, Image

import dlib_live
from retinaface import Retinaface
import os


class Database:
    def __init__(self):
        self.conn = pymysql.connect(host="127.0.0.1",
                                    port=3306,
                                    user="root",
                                    password="1300982918",
                                    database="rlsb")
        self.cursor = self.conn.cursor()

    def insert_name(self, name):
        query = "INSERT INTO name_table (name) VALUES (%s)"
        self.cursor.execute(query, (name,))
        self.conn.commit()

    def query_name(self):
        query = "SELECT name FROM name_table"
        self.cursor.execute(query)
        return [row[0] for row in self.cursor.fetchall()]

    def query_record(self):
        query = "SELECT * FROM record_table"
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def insert_record(self, username):
        import datetime
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        insert_query = "INSERT INTO record_table (name, check_time) VALUES (%s, %s)"
        self.cursor.execute(insert_query, (username, current_time))
        self.conn.commit()

    def close(self):
        self.cursor.close()
        self.conn.close()

def makeDir():
    if not os.path.exists("face_dataset"):
        os.mkdir("face_dataset")
    if not os.path.exists("FaceData"):
        os.mkdir("FaceData")

def getFace(name):
    cap = cv2.VideoCapture(0)
    num = 0
    while cap.isOpened():
        ret_flag, Vshow = cap.read()  # 得到每帧图像
        cv2.imshow("Capture_Test", Vshow)  # 显示图像
        k = cv2.waitKey(1) & 0xFF  # 按键判断
        if k == ord('s'):  # 保存
            cv2.imwrite("face_dataset/" + name + "_" + str(num) + ".jpg", Vshow)
            print("success to save" + str(num) + ".jpg")
            print("-------------------")
            num += 1
        elif k == ord(' '):  # 退出
            break
    #释放摄像头
    cap.release()
    #释放内存
    cv2.destroyAllWindows()


def encode_faces():
    '''
    在更换facenet网络后一定要重新进行人脸编码，运行encoding.py。
    '''
    retinaface = Retinaface(1)

    list_dir = os.listdir("face_dataset")
    image_paths = []
    names = []
    user_db = Database()
    for name in list_dir:
        image_paths.append("face_dataset/" + name)
        names.append(name.split("_")[0])
        # print(name.split("_")[0],names)
        try:
            user_db.insert_name(name.split("_")[0])
        except Exception as e:
            print(e)

    retinaface.encode_face_dataset(image_paths, names)
    user_db.close()
    return "Encoding complete!"

def add_face(name, names):
    makeDir()
    getFace(name)
    encode_faces()

class APP:
    def __init__(self):

        self.root = Tk()
        self.root.title('FACE')
        self.root.geometry('%dx%d' % (400, 300))

        # 创建数据库连接
        self.conn = pymysql.connect(host="127.0.0.1",
                                    port=3306,
                                    user="root",
                                    password="1300982918",
                                    database="rlsb")

        # 获取数据库游标
        self.cursor = self.conn.cursor()

        self.createFirstPage()

        # 新录入的人的姓名
        self.name = StringVar()

        mainloop()

    def createFirstPage(self):
        self.page1 = Frame(self.root)
        self.page1.grid()
        Label(self.page1, height=4, text='人脸识别系统', font=('粗体', 20)).grid(columnspan=2)

        # 获取用户名字列表
        self.usernames = self.query_name()

        self.button11 = Button(self.page1, width=18, height=2, text="签到打卡", bg='red', font=("宋", 12),
                               relief='raise', command=lambda: self.check(self.usernames))
        self.button11.grid(row=1, column=0, padx=25, pady=10)
        self.button12 = Button(self.page1, width=18, height=2, text="录入新的人脸", bg='green', font=("宋", 12),
                               relief='raise', command=self.createSecondPage)
        self.button12.grid(row=1, column=1, padx=25, pady=10)
        self.button13 = Button(self.page1, width=18, height=2, text="查询签到信息", bg='white', font=("宋", 12),
                               relief='raise', command=self.checkDataView)
        self.button13.grid(row=2, column=0, padx=25, pady=10)
        self.button14 = Button(self.page1, width=18, height=2, text="退出系统", bg='gray', font=("宋", 12),
                               relief='raise', command=self.quitMain)
        self.button14.grid(row=2, column=1, padx=25, pady=10)

    def createSecondPage(self):
        self.page1.grid_forget()
        self.page2 = Frame(self.root)
        self.page2.pack()
        Label(self.page2, text='欢迎使用人脸识别系统', font=('粗体', 20)).pack()

        font1 = ('宋', 18)
        self.text = Entry(self.page2, textvariable=self.name, width=20, font=font1)
        self.text.pack(side=LEFT)
        self.name.set('请输入姓名')

        self.button21 = Button(self.page2, text='确认', bg='white', font=("宋", 12),
                               relief='raise', command=lambda: add_face(self.name.get(), self.usernames))
        self.button21.pack(side=LEFT, padx=5, pady=10)

        self.button22 = Button(self.page2, text="返回", bg='white', font=("宋", 12),
                               relief='raise', command=self.p2back)
        self.button22.pack(side=LEFT, padx=10, pady=10)

    def checkDataView(self):
        self.page3 = Frame(self.root)
        self.page1.grid_forget()
        self.root.geometry('700x360')
        self.page3.pack()
        Label(self.page3, text='今日签到信息', bg='white', fg='red', font=('宋体', 25)).pack(side=TOP, fill='x')

        self.checkDate = ttk.Treeview(self.page3, show='headings', column=('id', 'name', 'record_time'))
        self.checkDate.column('id', width=100, anchor="center")
        self.checkDate.column('name', width=200, anchor="center")
        self.checkDate.column('record_time', width=300, anchor="center")

        self.checkDate.heading('id', text='签到序号')
        self.checkDate.heading('name', text='名字')
        self.checkDate.heading('record_time', text='签到时间')

        self.records = self.query_record()
        for i in self.records:
            self.checkDate.insert('', 'end', values=i)

        yscrollbar = Scrollbar(self.page3, orient=VERTICAL, command=self.checkDate.yview)
        self.checkDate.configure(yscrollcommand=yscrollbar.set)
        yscrollbar.pack(side=RIGHT, fill=Y)

        self.checkDate.pack(expand=1, fill=BOTH)

        Button(self.page3, width=20, height=2, text="返回", bg='gray', font=("宋", 12),
               relief='raise', command=self.p3back).pack(padx=20, pady=20)

    def p2back(self):
        self.page2.pack_forget()
        self.root.geometry('400x300')
        self.page1.grid()

    def p3back(self):
        self.root.geometry('400x300')
        self.page3.pack_forget()
        self.page1.grid()

    def quitMain(self):
        self.conn.close()  # 关闭数据库连接
        sys.exit(0)

    def query_name(self):
        query = "SELECT name FROM name_table"
        self.cursor.execute(query)
        result = self.cursor.fetchall()
        return [row[0] for row in result]

    def query_record(self):
        query = "SELECT id, name, record_time FROM record_table"
        self.cursor.execute(query)
        return self.cursor.fetchall()



    def check(self, names):
        detector = dlib_live.FaceDetection(0)  # 使用摄像头，也可以指定视频文件路径

        while True:
            flag = detector.get_flag()

            ref,frame = detector.process_frame()
            username= detector.get_name()
            if frame is None:
                break
            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if flag == 1:
                # 插入签到信息到数据库
                self.insert_record(username)
                print(flag)
                cv2.imwrite("last_frame.png", frame)
                # print(fname)
                break
        detector.release()
        cv2.destroyAllWindows()


    def insert_record(self, username):
        insert_query = "INSERT INTO record_table (name, record_time) VALUES (%s, %s)"
        current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        self.cursor.execute(insert_query, (username, current_time))
        self.conn.commit()

if __name__ == '__main__':
    demo = APP()
