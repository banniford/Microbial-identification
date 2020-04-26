import socket #通信模块
import sys #系统模块
import datetime as d #时间模块
import serial #串口通信连接arduino模块
ser = serial.Serial("/dev/ttyUSB0", 9600)#打开9600通信