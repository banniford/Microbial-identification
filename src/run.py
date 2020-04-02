import sys
import Microbial
from PyQt5.QtWidgets import  QApplication,QMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
import time
#
class fnc(Microbial.Ui_MainWindow):
    def __init__(self,MainWindow):
        super().setupUi(MainWindow)
        self.pushButton_7.clicked.connect(self.image)
        self.pushButton.clicked.connect(self.headline)
        self.pushButton_8.clicked.connect(self.close)
        self.progressBar.setHidden(True)
        self.pushButton_9.clicked.connect(self.collect)

    def headline(self):
        self.pushButton.setEnabled(False)
        self.label.setText("摄像头已连接")


    def image(self):
        pixmap = QPixmap('C:\\Users\some\Desktop\（毕设）全自动捕捉显微镜\深度学习资料\pyqt\\ui\image\\aaqa.jpg')
        self.label_2.setPixmap(pixmap)
        self.pushButton_7.setEnabled(False)
        self.pushButton_8.setEnabled(True)
        self.pushButton_2.setEnabled(True)
        self.pushButton_3.setEnabled(True)
        self.pushButton_4.setEnabled(True)
        self.pushButton_5.setEnabled(True)
        self.pushButton_6.setEnabled(True)
        self.pushButton_9.setEnabled(True)
        self.spinBox.setEnabled(True)

    def close(self, MainWindow):
        self.pushButton_7.setEnabled(True)
        self.pushButton_8.setEnabled(False)
        self.pushButton_2.setEnabled(False)
        self.pushButton_3.setEnabled(False)
        self.pushButton_4.setEnabled(False)
        self.pushButton_5.setEnabled(False)
        self.pushButton_6.setEnabled(False)
        self.pushButton_9.setEnabled(False)
        self.spinBox.setEnabled(False)
        self.label_2.setPixmap(QPixmap(None))
        self.label_2.setText("没有图像")

    def collect(self):
        self.progressBar.setHidden(False)
        self.progressBar.setRange(0,5)
        for i in range(6):
            self.progressBar.setValue(i)
            time.sleep(1)
            if i == 5 :
                self.progressBar.setHidden(True)






    # def fnc1(self):
    #     print("1")
    #     pixmap = QPixmap('C:\\Users\\some\\Desktop\\（毕设）全自动捕捉显微镜\\深度学习资料\\pyqt\\ui\\image\\jmj.jpg')
    #     print("2")
    #     self.label_2.setPixmap(pixmap)



if __name__ == "__main__":

    app = QApplication(sys.argv)

    mainWindow = QMainWindow()
    ui = fnc(mainWindow)#注意把类名修改为fnc

    # ui = Microbial.Ui_MainWindow()
    #
    # ui.setupUi(mainWindow)fnc类的构造函数已经调用了这个函数，这行代码可以删去
    mainWindow.show()

    sys.exit(app.exec_())