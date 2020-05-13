
import cv2
import numpy as np


from PyQt5.QtGui import QImage, QPixmap


class Video():

    def __init__(self, capture):
        self.capture = capture
        self.currentFrame = np.array([])

    def captureFrame(self):
        ret, readFrame = self.capture.read()
        return ret,readFrame

    def captureNextFrame(self):
        ret, readFrame = self.capture.read()
        if (ret == True):
            self.currentFrame=cv2.resize(readFrame, (800, 600))
            self.currentFrame = cv2.cvtColor(self.currentFrame, cv2.COLOR_BGR2RGB)

    def convertFrame(self):
        try:
            # 将视频转换为qt能接受的图片格式
            height, width = self.currentFrame.shape[:2]
            img = QImage(self.currentFrame.data, width, height, QImage.Format_RGB888)
            #将QIamg对象转为QPixmap对象
            img = QPixmap.fromImage(img)
            return img
        except:
            return None


# a,b=Video(cv2.VideoCapture(0)).captureFrame()
# print(a,b)

