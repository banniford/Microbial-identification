import sys
import Microbial
import parameter
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QFileDialog
from PyQt5.QtGui import QIntValidator,QDoubleValidator,QRegExpValidator
from PyQt5.QtGui import *
from PyQt5.QtCore import QRegExp
from utils.config import Config


class fnc(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.main_ui=Microbial.Ui_MainWindow()
        self.main_ui.setupUi(self)
        self.main_ui.pushButton_7.clicked.connect(self.image)
        self.main_ui.pushButton.clicked.connect(self.headline)
        self.main_ui.pushButton_8.clicked.connect(self.close)
        self.main_ui.label_4.setText("模型路径："+ Config["migrate_path"])
        self.main_ui.label_5.setText("模型路径："+ Config["model_path"])
        self.main_ui.pushButton_10.clicked.connect(self.open_image)
        self.main_ui.pushButton_11.clicked.connect(self.module)
        self.main_ui.pushButton_14.clicked.connect(self.migrate)

    def headline(self):
        self.main_ui.pushButton.setEnabled(False)
        self.main_ui.label.setText("摄像头已连接")


    def image(self):
        # 摄像头获取
        pixmap = QPixmap('C:\\Users\some\Desktop\（毕设）全自动捕捉显微镜\深度学习资料\pyqt\\ui\image\\aaqa.jpg')
        self.main_ui.label_2.setPixmap(pixmap)
        self.main_ui.pushButton_7.setEnabled(False)
        self.main_ui.pushButton_8.setEnabled(True)
        self.main_ui.pushButton_2.setEnabled(True)
        self.main_ui.pushButton_3.setEnabled(True)
        self.main_ui.pushButton_4.setEnabled(True)
        self.main_ui.pushButton_5.setEnabled(True)
        self.main_ui.pushButton_6.setEnabled(True)
        self.main_ui.pushButton_9.setEnabled(True)

    def close(self):
        self.main_ui.pushButton_7.setEnabled(True)
        self.main_ui.pushButton_8.setEnabled(False)
        self.main_ui.pushButton_2.setEnabled(False)
        self.main_ui.pushButton_3.setEnabled(False)
        self.main_ui.pushButton_4.setEnabled(False)
        self.main_ui.pushButton_5.setEnabled(False)
        self.main_ui.pushButton_6.setEnabled(False)
        self.main_ui.pushButton_9.setEnabled(False)
        self.main_ui.label_2.setPixmap(QPixmap(None))
        self.main_ui.label_2.setText("没有图像")

    def open_image(self):
        Fname,_=QFileDialog.getOpenFileName(self,"打开文件",".","图像文件(*.jpg *.png)")
        if Fname != "":
            self.main_ui.label_3.setPixmap(QPixmap(Fname))
            self.main_ui.pushButton_12.setEnabled(True)

    def migrate(self):
        Fname, _ = QFileDialog.getOpenFileName(self, "打开文件", ".", "图像文件(*.pth)")
        if Fname!="":
            self.main_ui.label_5.setText("模型路径："+ Fname)
            Config["migrate_path"]=Fname

    def module(self):
        Fname, _ = QFileDialog.getOpenFileName(self, "打开文件", ".", "图像文件(*.pth)")
        if Fname!="":
            self.main_ui.label_5.setText("模型路径："+ Fname)
            Config["module_path"]=Fname

class parameters(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.configtext=[]
        self.widget_ui=parameter.Ui_Form()
        self.widget_ui.setupUi(self)
        self.widget_ui.label_5.setText("min_dim：%s" % (Config["min_dim"]))
        self.widget_ui.label_6.setText("confidence：%s" % (Config["confidence"]))
        self.widget_ui.label_7.setText("Cuda：%s" % (Config["Cuda"]))
        self.widget_ui.label_8.setText("Epoch：%s" % (Config["Epoch"]))
        self.widget_ui.label_9.setText("trainval_percent：%s" % (Config["trainval_percent"]))
        self.widget_ui.label_10.setText("train_percent：%s" % (Config["train_percent"]))
        self.widget_ui.label_11.setText("Batch_size：%s" % (Config["Batch_size"]))
        self.widget_ui.label_12.setText("lr：%s" % (Config["lr"]))
        self.widget_ui.lineEdit.setPlaceholderText("100-1000")
        self.widget_ui.lineEdit_2.setPlaceholderText("0-0.9")
        self.widget_ui.lineEdit_3.setPlaceholderText("1-500")
        self.widget_ui.lineEdit_4.setPlaceholderText("0-1")
        self.widget_ui.lineEdit_5.setPlaceholderText("0-1")
        self.widget_ui.lineEdit_6.setPlaceholderText("8,16,32,64")
        self.widget_ui.lineEdit_7.setPlaceholderText("0-1")
        self.widget_ui.pushButton.clicked.connect(self.right)
        self.widget_ui.pushButton.clicked.connect(self.close)
        # 校验
        self.widget_ui.lineEdit.editingFinished.connect(self.enter_line)
        self.widget_ui.lineEdit_3.editingFinished.connect(self.enter_line)
        self.widget_ui.lineEdit_6.editingFinished.connect(self.enter_line)
        self.widget_ui.lineEdit_2.editingFinished.connect(self.enter_line)
        self.widget_ui.lineEdit_4.editingFinished.connect(self.enter_line)
        self.widget_ui.lineEdit_5.editingFinished.connect(self.enter_line)
        self.widget_ui.lineEdit_7.editingFinished.connect(self.enter_line)

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
        if self.widget_ui.lineEdit.text() != "" and not(self.widget_ui.lineEdit.text().isdigit()):
            self.widget_ui.lineEdit.setText(str(Config["min_dim"]))
        if self.widget_ui.lineEdit.text() !="" and self.widget_ui.lineEdit.text().isdigit():
            edit1=int(self.widget_ui.lineEdit.text())
            if edit1<100:
                self.widget_ui.lineEdit.setText("100")
            elif edit1>1000:
                self.widget_ui.lineEdit.setText("1000")

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



    def right(self):

        if self.widget_ui.lineEdit.text() != "":
            # line1 = self.widget_ui.lineEdit.text()
            # print(line1)
            Config["min_dim"]=int(self.widget_ui.lineEdit.text())
            self.widget_ui.label_5.setText("min_dim：%s" % (Config["min_dim"]))

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

        if self.widget_ui.textEdit.toPlainText()!="":
            classes=self.widget_ui.textEdit.toPlainText().split(" ")
            with open("neural_network/model_data/voc_classes.txt","w")as f:
                for i in classes:
                    if i !='':
                        f.write(i)
                        f.write("\n")
            classes=[]
            # for i in classes:
            #     if i =="":
            #         classes.remove(i)
            # self.configtext=classes remove删除会使索引向前进一，会跳过一个索引
            with open("neural_network/model_data/voc_classes.txt", "r")as f:
                for line in f.readlines():  # 依次读取每行
                    line = line.strip()  # 去掉每行头尾空白
                    if line != '' and line not in classes:
                        classes.append(line)
            self.configtext = classes
        else:
            classes=[]
            with open("neural_network/model_data/voc_classes.txt", "r")as f:
                for line in f.readlines():  # 依次读取每行
                    line = line.strip()  # 去掉每行头尾空白
                    if line != '' and line not in classes:
                        classes.append(line)
            self.configtext = classes





    # def fnc1(self):
    #     print("1")
    #     pixmap = QPixmap('C:\\Users\\some\\Desktop\\（毕设）全自动捕捉显微镜\\深度学习资料\\pyqt\\ui\\image\\jmj.jpg')
    #     print("2")
    #     self.label_2.setPixmap(pixmap)

def interaction():
    configtext=("类别："+str(widget.configtext)+" | "+"min_dim：%s" % (Config["min_dim"])+" | "+"Epoch：%s" % (Config["Epoch"])
                +" | "+"Cuda:%s"%(Config["Cuda"])+" | "+ "Batch_size：%s" % (Config["Batch_size"])+" | "+"confidence：%s" % (Config["confidence"])+" | "+
                "trainval_percent：%s" % (Config["trainval_percent"])+" | "+"train_percent：%s" % (Config["train_percent"])+
                " | "+"lr：%s" % (Config["lr"]))
    # toPlainText 获取 setText 设置 setPlainText 多行设置 append 追加
    window.main_ui.textEdit.append(str(configtext))
    num_class=len(widget.configtext)
    Config["num_classes"]=num_class+1



if __name__ == "__main__":

    app = QApplication(sys.argv)
    window=fnc()
    widget=parameters()
    btn=window.main_ui.pushButton_13
    btn.clicked.connect(widget.show)
    btn1=widget.widget_ui.pushButton
    btn1.clicked.connect(interaction)


    window.show()

    sys.exit(app.exec_())