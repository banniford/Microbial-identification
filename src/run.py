import sys
sys.path.append(sys.path[0]+"\\neural_network") # 添加路径 导包，下面红线不用担心
import cv2
import Microbial
import parameter
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QFileDialog
from PyQt5.QtGui import QRegExpValidator, QPixmap
from PyQt5.QtCore import QRegExp, QTimer

from nets.ssd import get_ssd
from utils.config import Config
from neural_network.utils.voc_annotation import voc
from neural_network.predict import SSD
from neural_network.train import train
from PIL import Image
from mid_module import summary
from arduino_control import Video

import serial # 串口通信连接arduino模块



class fnc(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.main_ui=Microbial.Ui_MainWindow()
        self.main_ui.setupUi(self)
        self.img = ""
        self.train_start = train()
        self.train_start.msg.connect(self.train_msg)
        self.main_ui.pushButton_7.clicked.connect(self.control)
        self.main_ui.pushButton_8.clicked.connect(self.close_control)
        self.main_ui.pushButton_9.clicked.connect(self.collect)
        self.conunt=0
        self.main_ui.label_4.setText("模型路径："+ Config["migrate_path"])
        self.main_ui.label_5.setText("模型路径："+ Config["model_path"])
        self.main_ui.pushButton_10.clicked.connect(self.open_image)
        self.main_ui.pushButton_11.clicked.connect(self.module)
        self.main_ui.pushButton_14.clicked.connect(self.migrate)
        self.main_ui.pushButton_12.clicked.connect(self.predict)
        self.main_ui.pushButton_15.clicked.connect(self.check)
        self.main_ui.pushButton_16.clicked.connect(self.train)
        self.main_ui.pushButton_17.clicked.connect(self.stop_train)
        self.main_ui.pushButton_18.clicked.connect(self.structure)

        self.main_ui.pushButton_2.clicked.connect(self.up_Z)
        self.main_ui.pushButton_3.clicked.connect(self.left)
        self.main_ui.pushButton_4.clicked.connect(self.right)
        self.main_ui.pushButton_5.clicked.connect(self.down_Z)
        self.main_ui.pushButton_6.clicked.connect(self.auto_control)
        self.main_ui.pushButton_20.clicked.connect(self.up_Y)
        self.main_ui.pushButton_19.clicked.connect(self.down_Y)

        self.timer_camera =QTimer(self)  # 初始化定时器
        # self.cap = cv2.VideoCapture()  # 初始化摄像头
        self.video = Video(cv2.VideoCapture(0))
        self.timer_camera.timeout.connect(self.show_camera)
        self.main_ui.pushButton.clicked.connect(self.headline)

    def show_camera(self):
        self.video.captureNextFrame()
        self.Frame_img=self.video.convertFrame()
        self.main_ui.label_2.setPixmap(self.Frame_img)



    def headline(self):
        if self.timer_camera.isActive() == False:
            flag=self.video.capture.open(0)
            if flag == False:
                self.main_ui.textEdit.append("检测不到摄像头")
            else:
                self.timer_camera.start(30)
                self.main_ui.pushButton_9.setEnabled(True)
                self.main_ui.label.setText("摄像头已连接")
                self.main_ui.pushButton.setText('关闭摄像头')
        else:
            self.timer_camera.stop()
            self.video.capture.release()
            self.main_ui.pushButton_9.setEnabled(False)
            self.main_ui.label_2.setPixmap(QPixmap(None))
            self.main_ui.label_2.setText("没有图像")
            self.main_ui.pushButton.setText('打开摄像头')
            self.main_ui.label.setText("摄像头未连接")

        #摄像头检测和连接
        # try:
        #     self.video = Video(cv2.VideoCapture(0))
        #     self.video.captureNextFrame()
        #     self.main_ui.label_2.setPixmap(self.video.convertFrame())
        # except:
        #     self.main_ui.textEdit.append("检测不到摄像头")
        # else:
        #     self.main_ui.pushButton_9.setEnabled(True)
        #     self.main_ui.pushButton.setEnabled(False)
        #     self.main_ui.label.setText("摄像头已连接")


    def collect(self):
        self.conunt+=1
        self.Frame_img.save("neural_network/VOCdevkit/VOC2007/JPEGImages/%s.jpg"%(self.conunt))


    def structure(self):
        model = get_ssd("train", Config["num_classes"])
        self.structure_start = summary()
        self.structure_start.msg.connect(self.structure_msg)
        self.structure_start.summary(model, input_size=(3, Config["min_dim"],  Config["min_dim"]),batch_size=Config["Batch_size"])

    def structure_msg(self,msg):
        self.main_ui.textEdit_2.append(msg)


    def stop_train(self):
        #中止训练
        self.main_ui.pushButton_13.setEnabled(True)
        self.main_ui.pushButton_14.setEnabled(True)
        self.main_ui.pushButton_15.setEnabled(True)
        self.main_ui.pushButton_18.setEnabled(True)
        self.main_ui.pushButton_16.setEnabled(False)
        self.main_ui.pushButton_17.setEnabled(False)
        self.main_ui.textEdit.append("正在退出......请稍等")
        self.train_start.flag=False


    def predict_msg(self,msg):
        self.main_ui.textEdit.append(msg)


    def train_msg(self,msg):
        self.main_ui.textEdit.append(msg)

    def train(self):
        # 多线程
        self.main_ui.pushButton_13.setEnabled(False)
        self.main_ui.pushButton_14.setEnabled(False)
        self.main_ui.pushButton_15.setEnabled(False)
        self.main_ui.pushButton_18.setEnabled(False)
        self.main_ui.pushButton_17.setEnabled(True)
        self.train_start.flag = True
        self.main_ui.textEdit.append("训练开始......")
        try:
            self.train_start.start()
        except:
            self.train_start.flag = False
            self.main_ui.textEdit.append("训练参数有误")

    def predict(self):

        self.main_ui.textEdit.append("开始预测......")
        img = self.img
        try:
            image = Image.open(img)
            # image.resize((640, 480), Image.ANTIALIAS)
        except:
            self.main_ui.textEdit.append("图片损坏")
        else:
            self.predict_start.image=image
            self.predict_start.start()

    def control(self):
        # 连接arduino
        try:
            self.ser = serial.Serial("/dev/ttyUSB0", 9600)
        except:
            self.main_ui.textEdit.append("无法连接arduino")
        else:
            self.main_ui.pushButton_7.setEnabled(False)
            self.main_ui.pushButton_8.setEnabled(True)
            self.main_ui.pushButton_2.setEnabled(True)
            self.main_ui.pushButton_3.setEnabled(True)
            self.main_ui.pushButton_4.setEnabled(True)
            self.main_ui.pushButton_5.setEnabled(True)
            self.main_ui.pushButton_6.setEnabled(True)


    def close_control(self):
        self.main_ui.pushButton_7.setEnabled(True)
        self.main_ui.pushButton_8.setEnabled(False)
        self.main_ui.pushButton_2.setEnabled(False)
        self.main_ui.pushButton_3.setEnabled(False)
        self.main_ui.pushButton_4.setEnabled(False)
        self.main_ui.pushButton_5.setEnabled(False)
        self.main_ui.pushButton_6.setEnabled(False)
        # self.main_ui.label_2.setPixmap(QPixmap(None))
        # self.main_ui.label_2.setText("没有图像")

    def up_Y(self):
        str1="a"
        self.ser.write(str1.encode('utf-8'))


    def down_Y(self):
        str1 ="b"
        self.ser.write(str1.encode('utf-8'))
        self.ser.flushInput()

    def up_Z(self):
        str1="i"
        self.ser.write(str1.encode('utf-8'))
        self.ser.flushInput()


    def down_Z(self):
        str1 = "j"
        self.ser.write(str1.encode('utf-8'))
        self.ser.flushInput()

    def left(self):
        str1 = "e"
        self.ser.write(str1.encode('utf-8'))
        self.ser.flushInput()


    def right(self):
        str1 = "f"
        self.ser.write(str1.encode('utf-8'))
        self.ser.flushInput()


    def auto_control(self):
        i = 0
        a = 'j'
        b = 'l'
        c = 'n'
        d = 'm'
        definition = self.variance_of_laplacian(self.Frame_img)
        if not definition:
            self.main_ui.textEdit.append("图片清晰度出错")
        else:
            if float(definition) < 10.0:
                i = i + 1
                print(i)
                if i > 10.0:
                    print(definition)
                    self.ser.write(a.encode('utf-8'))  # 向串口写‘a’，arduino接受
                    self.ser.flushInput()  # 清除缓存器
                    i = 0
            elif 10.0 <= float(definition) <= 20.0:
                i = i + 1
                if i > 10:
                    print(definition)
                    self.ser.write(a.encode('utf-8'))
                    self.ser.flushInput()
                    i = 0
            elif 20.0 <= float(definition) <= 30.0:
                i = i + 1
                if i > 7:
                    print(definition)
                    self.ser.write(b.encode('utf-8'))
                    self.ser.flushInput()
                    i = 0
            elif 30.0 <= float(definition) < 50.0:
                i = i + 1
                if i > 5:
                    print(definition)
                    self.ser.write(c.encode('utf-8'))
                    self.ser.flushInput()
                    i = 0
            elif float(definition) > 50.0:
                print(definition)
                self.main_ui.textEdit.append("完成对焦")
                self.ser.write(d.encode('utf-8'))
                self.ser.flushInput()
                self.ser.close()  # 关闭端口


    # img = cv2.imread('neural_network/img/1.jpg')
    # 清晰度判断
    # img = cv2.imread('neural_network/img/1.jpg')
    # d = self.variance_of_laplacian(img)
    # self.main_ui.textEdit.append(str(d))
    def variance_of_laplacian(self,image):
        return cv2.Laplacian(image, cv2.CV_64F).var()


    def open_image(self):
        Fname,_=QFileDialog.getOpenFileName(self,"打开文件",".","图像文件(*.jpg *.png)")
        self.main_ui.textEdit.append("图片路径为:"+Fname)
        if Fname != "":
            self.main_ui.label_3.setPixmap(QPixmap(Fname))
            self.img=Fname
            self.main_ui.pushButton_12.setEnabled(True)

    def migrate(self):
        Fname, _ = QFileDialog.getOpenFileName(self, "打开文件", ".", "模型文件(*.pth)")
        if Fname!="":
            self.main_ui.label_4.setText("模型路径："+ Fname)
            Config["migrate_path"]=Fname


    def module(self):
        Fname, _ = QFileDialog.getOpenFileName(self, "打开文件", ".", "模型文件(*.pth)")
        if Fname != "":
            self.main_ui.label_5.setText("模型路径：" + Fname)
            Config["model_path"] = Fname
            try:
                self.predict_start = SSD()
            except:
                self.main_ui.textEdit.append("模型中类别不匹配，请更换模型或更改类别参数")
                self.main_ui.pushButton_10.setEnabled(False)
                self.main_ui.pushButton_12.setEnabled(False)
            else:
                self.predict_start.set_image.connect(self.set_image)
                self.predict_start.msg.connect(self.predict_msg)
                self.main_ui.label_5.setText("模型路径：" + Config["model_path"])
                self.main_ui.pushButton_10.setEnabled(True)

    def set_image(self,path):
        if path=="":
            self.main_ui.label_3.setPixmap(QPixmap(self.img))
        else:
            self.main_ui.label_3.setPixmap(QPixmap(path))




    def check(self):
        # ！！！
        voc()
        self.main_ui.textEdit.append("校验完成")
        self.main_ui.pushButton_16.setEnabled(True)




class parameters(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.configtext=[]
        self.widget_ui=parameter.Ui_Form()
        self.widget_ui.setupUi(self)
        self.widget_ui.label_6.setText("confidence：%s" % (Config["confidence"]))
        self.widget_ui.label_7.setText("Cuda：%s" % (Config["Cuda"]))
        self.widget_ui.label_8.setText("Epoch：%s" % (Config["Epoch"]))
        self.widget_ui.label_9.setText("trainval_percent：%s" % (Config["trainval_percent"]))
        self.widget_ui.label_10.setText("train_percent：%s" % (Config["train_percent"]))
        self.widget_ui.label_11.setText("Batch_size：%s" % (Config["Batch_size"]))
        self.widget_ui.label_12.setText("lr：%s" % (Config["lr"]))
        self.widget_ui.label_2.setText("loc_loss：%s" % (Config["loc_loss"]))
        self.widget_ui.label_3.setText("conf_loss：%s" % (Config["conf_loss"]))
        self.widget_ui.lineEdit_2.setPlaceholderText("0-0.9")
        self.widget_ui.lineEdit_3.setPlaceholderText("1-500")
        self.widget_ui.lineEdit_4.setPlaceholderText("0-1")
        self.widget_ui.lineEdit_5.setPlaceholderText("0-1")
        self.widget_ui.lineEdit_6.setPlaceholderText("8,16,32,64")
        self.widget_ui.lineEdit_7.setPlaceholderText("0-1")
        self.widget_ui.lineEdit_8.setPlaceholderText("0-2")
        self.widget_ui.lineEdit_9.setPlaceholderText("0-5")
        self.widget_ui.pushButton.clicked.connect(self.right)
        self.widget_ui.pushButton.clicked.connect(self.close)
        self.widget_ui.pushButton_2.clicked.connect(self.close)
        # 校验
        self.widget_ui.lineEdit_3.editingFinished.connect(self.enter_line)
        self.widget_ui.lineEdit_6.editingFinished.connect(self.enter_line)
        self.widget_ui.lineEdit_2.editingFinished.connect(self.enter_line)
        self.widget_ui.lineEdit_4.editingFinished.connect(self.enter_line)
        self.widget_ui.lineEdit_5.editingFinished.connect(self.enter_line)
        self.widget_ui.lineEdit_7.editingFinished.connect(self.enter_line)
        self.widget_ui.lineEdit_8.editingFinished.connect(self.enter_line)
        self.widget_ui.lineEdit_9.editingFinished.connect(self.enter_line)

        # 字符和数字表示
        # reg=QRegExp("[a-zA-Z0-9]+$")
        # 正浮点数
        # reg = QRegExp("^(([0-9]+\.[0-9]*[1-9][0-9]*)|([0-9]*[1-9][0-9]*\.[0-9]+)|([0-9]*[1-9][0-9]*))$")
        reg = QRegExp("^\d+(\.\d+)?$")
        validator=QRegExpValidator(self)
        validator.setRegExp(reg)
        # #设置校验器
        self.widget_ui.lineEdit_2.setValidator(validator)
        self.widget_ui.lineEdit_4.setValidator(validator)
        self.widget_ui.lineEdit_5.setValidator(validator)
        #
        self.widget_ui.lineEdit_7.setValidator(validator)

    def enter_line(self):

        if self.widget_ui.lineEdit_3.text() != "" and not(self.widget_ui.lineEdit_3.text().isdigit()):
            self.widget_ui.lineEdit_3.setText(str(Config["Epoch"]))
        if self.widget_ui.lineEdit_3.text() != "" and self.widget_ui.lineEdit_3.text().isdigit():
            edit3=int(self.widget_ui.lineEdit_3.text())
            if edit3<1:
                self.widget_ui.lineEdit_3.setText("1")
            elif edit3>500:
                self.widget_ui.lineEdit_3.setText("500")

        if self.widget_ui.lineEdit_6.text() != "" and not(self.widget_ui.lineEdit_6.text().isdigit()):
            self.widget_ui.lineEdit_6.setText(str(Config["Batch_size"]))
        if self.widget_ui.lineEdit_6.text() != "" and self.widget_ui.lineEdit_6.text().isdigit():
            edit6 = int(self.widget_ui.lineEdit_6.text())
            if edit6<2:
                self.widget_ui.lineEdit_6.setText("2")
            elif edit6>1024:
                self.widget_ui.lineEdit_6.setText("1024")

        # if self.widget_ui.lineEdit_2.text() != "" and not(self.widget_ui.lineEdit_2.text().count('.') ==1):
        #     self.widget_ui.lineEdit_2.setText(str(Config["confidence"]))
        if self.widget_ui.lineEdit_2.text() != "" :
            edit2 = float(self.widget_ui.lineEdit_2.text())
            if edit2<0:
                self.widget_ui.lineEdit_2.setText("0")
            elif edit2>0.9:
                self.widget_ui.lineEdit_2.setText("0.9")

        if self.widget_ui.lineEdit_4.text() != "" :
            edit4 = float(self.widget_ui.lineEdit_4.text())
            if edit4<=0:
                self.widget_ui.lineEdit_4.setText("0.1")
            elif edit4>=1:
                self.widget_ui.lineEdit_4.setText("1")

        if self.widget_ui.lineEdit_5.text() != "":
            edit5 = float(self.widget_ui.lineEdit_5.text())
            if edit5<=0:
                self.widget_ui.lineEdit_5.setText("0.1")
            elif edit5>=1:
                self.widget_ui.lineEdit_5.setText("1")

        if self.widget_ui.lineEdit_7.text() != "":
            edit7 = float(self.widget_ui.lineEdit_7.text())
            if edit7<=0:
                self.widget_ui.lineEdit_7.setText(str(Config["lr"]))
            elif edit7>=1:
                self.widget_ui.lineEdit_7.setText(str(Config["lr"]))


        if self.widget_ui.lineEdit_8.text() != "" :
            edit8 = float(self.widget_ui.lineEdit_8.text())
            if edit8<=0:
                self.widget_ui.lineEdit_8.setText("0.5")
            elif edit8>=2:
                self.widget_ui.lineEdit_8.setText("2")


        if self.widget_ui.lineEdit_9.text() != "":
            edit9 = float(self.widget_ui.lineEdit_9.text())
            if edit9<=0:
                self.widget_ui.lineEdit_9.setText("0.5")
            elif edit9 >=5:
                self.widget_ui.lineEdit_9.setText("5")



    def right(self):


        if self.widget_ui.radioButton.isChecked()==True:
            Config["Cuda"]=True

        if self.widget_ui.lineEdit_3.text() != "":
            Config["Epoch"]=int(self.widget_ui.lineEdit_3.text())
            self.widget_ui.label_8.setText("Epoch：%s" % (Config["Epoch"]))

        if self.widget_ui.lineEdit_6.text() != "":
            Config["Batch_size"]=int(self.widget_ui.lineEdit_6.text())
            self.widget_ui.label_11.setText("Batch_size：%s" % (Config["Batch_size"]))

        if self.widget_ui.lineEdit_2.text() != "":
            Config["confidence"]=float(self.widget_ui.lineEdit_2.text())
            self.widget_ui.label_6.setText("confidence：%s" % (Config["confidence"]))

        if self.widget_ui.lineEdit_4.text() != "":
            Config["trainval_percent"]=float(self.widget_ui.lineEdit_4.text())
            self.widget_ui.label_9.setText("trainval_percent：%s" % (Config["trainval_percent"]))

        if self.widget_ui.lineEdit_5.text() != "":
            Config["train_percent"]=float(self.widget_ui.lineEdit_5.text())
            self.widget_ui.label_10.setText("train_percent：%s" % (Config["train_percent"]))

        if self.widget_ui.lineEdit_7.text() != "":
            Config["lr"]=float(self.widget_ui.lineEdit_7.text())
            self.widget_ui.label_12.setText("lr：%s" % (Config["lr"]))

        if self.widget_ui.lineEdit_8.text() != "":
            Config["loc_loss"]=float(self.widget_ui.lineEdit_8.text())
            self.widget_ui.label_2.setText("loc_loss：%s" % (Config["loc_loss"]))

        if self.widget_ui.lineEdit_9.text() != "":
            Config["conf_loss"]=float(self.widget_ui.lineEdit_9.text())
            self.widget_ui.label_3.setText("conf_loss：%s" % (Config["conf_loss"]))

        if self.widget_ui.textEdit.toPlainText()!="":
            classes=self.widget_ui.textEdit.toPlainText().split(" ")
            with open("neural_network/model_data/voc_classes.txt","w")as f:
                for i in classes:
                    if i !='' or i!="\n":
                        f.write(i)
                        f.write("\n")


def interaction():
    classes=[]
    with open("neural_network/model_data/voc_classes.txt", "r")as f:
        for line in f.readlines():  # 依次读取每行
            line = line.strip()  # 去掉每行头尾空白
            if line != '' and line not in classes:
                classes.append(line)
    configtext=("类别："+str(classes)+" | "+"min_dim：%s" % (Config["min_dim"])+" | "+"Epoch：%s" % (Config["Epoch"])
                +" | "+"Cuda:%s"%(Config["Cuda"])+" | "+ "Batch_size：%s" % (Config["Batch_size"])+" | "+"confidence：%s" % (Config["confidence"])+" | "+
                "trainval_percent：%s" % (Config["trainval_percent"])+" | "+"train_percent：%s" % (Config["train_percent"])+
                " | "+"lr：%s" % (Config["lr"])+" | "+"loc_loss：%s" % (Config["loc_loss"])+" | "+"conf_loss：%s" % (Config["conf_loss"]))
    # toPlainText 获取 setText 设置 setPlainText 多行设置 append 追加
    window.main_ui.textEdit.append(str(configtext))
    num_class=len(classes)
    Config["num_classes"]=num_class+1



if __name__ == "__main__":

    app = QApplication(sys.argv)
    window=fnc()
    widget=parameters()
    btn=window.main_ui.pushButton_13
    btn.clicked.connect(widget.show)
    btn1=widget.widget_ui.pushButton
    btn1.clicked.connect(interaction)
    btn2=widget.widget_ui.pushButton_2
    btn2.clicked.connect(interaction)


    window.show()

    sys.exit(app.exec_())