from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, QFileDialog
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5 import QtGui 
from PyQt5.QtGui import QPixmap
import DevUi
from Function import Function_UI

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import sys
import queue as Queue
from datetime import datetime
import serial, serial.tools.list_ports
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

flag_view = 0

# Load model for prediction
model = keras.models.load_model('normal-abnormal-100-0.8444-0.8300.h5')

class MatplotlibWidget(QWidget):
    def __init__(self, parent=None) :
        super(MatplotlibWidget, self).__init__(parent)
        self.figure = Figure(figsize=(10,10))
        self.canvas = FigureCanvasQTAgg(self.figure)
        #self.canvas = FigureCanvasQTAgg(fig)
        #self.axis = self.figure.add_subplot()
        #self.axis = self.figure.add_axes(rect, projection=None, polar=False, **kwargs)
        self.ax = self.figure.add_subplot()
        
        self.layoutvertical = QVBoxLayout(self)
        self.layoutvertical.addWidget(self.canvas)

    def update_plot(self, data):
        self.figure.clear()
        self.ax=self.figure.add_subplot()

        

class MainWidget(QWidget, DevUi.Ui_Footprint):
    def __init__(self):
        super(MainWidget, self).__init__()
        #self.main_w = QWidget()
        #self.uic = DevUi.Ui_Footprint()
        #self.uic.setupUi(self.main_w)

        self.setupUi(self)
        self.init_widget()

        self.serial = Function_UI()
        self.serialPort = serial.Serial()
        
        self.BaudrateList.addItems(self.serial.baudList)
        self.BaudrateList.setCurrentText('115200')
        self.update_ports()


        if flag_view == 0:
            self.plot_zero()
        elif flag_view == 1:
            self.plot_heatmap()
        #self.plot_home()
        self.ConnectButton.clicked.connect(self.f_connect)
        self.RefreshButton.clicked.connect(self.f_refresh)
        self.SaveButton.pressed.connect(self.f_save)
        self.ScanButton.pressed.connect(self.send_data)
        self.ProcessButton.pressed.connect(self.predict_data)
        self.serial.data_available.connect(self.update_view)

    def init_widget(self):
        self.matplotlibwidget = MatplotlibWidget()
        self.layoutvertical = QVBoxLayout(self.PlotMap)
        self.layoutvertical.addWidget(self.matplotlibwidget)

    def f_save(self):
        self.textDisplay.append("Capturing Footprint Image")
        now = datetime.now()
        print("save image")
        fig,ax=plt.subplots(figsize=(10,10))
        # ax.imshow(self.data_sensor,cmap="seismic")
        ax.imshow(self.data_sensor,cmap="Reds")
        # ax.imshow(self.data_sensor,cmap="seismic",vmin=0,vmax=4094)
        plt.savefig("FPIMG_"+now.strftime("%y%m%d_%H%M%S")+".png", dpi=300)
        self.textDisplay.append("Done Saving Image, Please Check")

    def f_refresh(self):
        self.flag_view = 0
        self.textDisplay.clear()
        #self.textDisplay.append("Device Disconnected")
        self.textDisplay.append("Device List Updated")
        #print(self.flag_view)
        self.plot_zero()
        #self.RefreshtButton.setEnabled(False)
        self.ConnectButton.setEnabled(True)
        self.SaveButton.setEnabled(False)
        self.ScanButton.setEnabled(False)
        self.update_ports()

    def f_connect(self):
        self.flag_view = 1
        
        #self.textDisplay.clear()
        #self.textDisplay.append("Device Connected")
        #self.RefreshButton.setEnabled(False)
        #self.ConnectButton.setEnabled(True)
        #self.SaveButton.setEnabled(True)
        self.f_connect_serial()


    def plot_zero(self):
        sensor1 = np.zeros((32,32))
        # self.matplotlibwidget.ax.imshow(sensor1,cmap="seismic")
        self.matplotlibwidget.ax.imshow(sensor1,cmap="Reds")
        self.matplotlibwidget.canvas.draw()

    def plot_map(self):
        #self.matplotlibwidget.axis.clear()
        self.sensor1 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1000, 1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1800, 0, 900, 2500, 3000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1600, 0, 1700, 0, 800, 2400, 2500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 400, 1600, 0, 800, 0, 750, 800, 700, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 2000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1500, 0, 0, 1000, 2000, 1800, 1800, 1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1000, 2500, 2500, 2000, 2500, 2500, 2600, 500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1000, 2500, 3050, 3050, 3200, 3250, 3050, 3200, 800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 3000, 3500, 3000, 3000, 3000, 3000, 3000, 3000, 1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 3000, 3050, 3000, 3000, 3000, 3000, 3000, 3000, 800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 2820, 3040, 3000, 3000, 3000, 3000, 3000, 3000, 800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 2800, 3060, 3000, 3000, 3000, 3000, 1050, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 3000, 3040, 3000, 3000, 3000, 1080, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 3000, 3000, 3000, 3000, 1500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 2800, 3000, 3000, 3000, 1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1050, 2800, 3000, 3000, 800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1020, 3000, 3000, 3000, 3000, 500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 2900, 3000, 3000, 3000, 3000, 800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 2800, 3000, 3000, 3000, 3000, 750, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 3000, 3250, 3000, 3000, 3000, 800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1000, 3200, 3500, 3000, 3250, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1080, 1080, 1000, 1020, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

        
        #while flag_view == 1 :
        print("connect tombol : ", self.flag_view)
        #heatmap=np.random.rand(32,32)
        
        
        fig,ax=plt.subplots(figsize=(10,10))
        # self.matplotlibwidget.ax.imshow(self.sensor1,cmap="seismic")
        self.matplotlibwidget.ax.imshow(self.sensor1,cmap="Reds")
        self.matplotlibwidget.canvas.draw()

        
    def update_view(self,data):
        self.textDisplay.append(data)
        #self.matplotlibwidget.figure.clear()
        # print("data diterima : ")
        # print(data)
        
        data_split = data.split(',')
        #data_split.pop()
        # print("data setelah split")
        data_split = [int(i) for i in data_split]
        # print(data_split)

        data_array = np.array(data_split)
        # print("data sensor int : ", type(data_array))
        # print(data_array)
        self.data_sensor = data_array.reshape(32,32)
        print("data reshape : ", type(self.data_sensor))
        print(self.data_sensor)
                
        fig,ax=plt.subplots(figsize=(10,10))
        #ax.imshow(self.sensor1,cmap="seismic")
        #plt.savefig('IMGfootprint.png',dpi=200)
        # self.matplotlibwidget.ax.imshow(self.data_sensor,cmap="seismic")
        self.matplotlibwidget.ax.imshow(self.data_sensor,cmap="Reds")
        # self.matplotlibwidget.ax.imshow(self.data_sensor,cmap="seismic",vmin=0,vmax=4094)
        self.matplotlibwidget.canvas.draw()


    def f_connect_serial(self):
        if(self.ConnectButton.isChecked()):
            port = self.PortList.currentText()
            baud = self.BaudrateList.currentText()
            self.serial.serialPort.port = port
            self.serial.serialPort.baudrate = baud
            self.serial.connect_serial()

            if(self.serial.serialPort.is_open):
                self.BaudrateList.setEnabled(False)
                self.PortList.setEnabled(False)
                self.RefreshButton.setEnabled(False)
                self.SaveButton.setEnabled(True)
                self.ProcessButton.setEnabled(True)
                self.ScanButton.setEnabled(True)
                self.ConnectButton.setText("Disconnect")
                self.textDisplay.clear()
                self.textDisplay.append("Device Connected")
                # self.send_data()
                #self.plot_map()
            else:
                self.BaudrateList.setEnabled(False)
                self.PortList.setEnabled(False)
                self.RefreshButton.setEnabled(False)
                self.ScanButton.setEnabled(False)
                self.ProcessButton.setEnabled(False)
                self.SaveButton.setEnabled(False)
                self.textDisplay.clear()
                self.textDisplay.append("Failed Built Connection !!")
                self.textDisplay.append("Silakan 'Diconnect' terlebih dahulu, Kemudian pilih 'PORT' & 'Baudrate' yang Valid, kemudian 'Connect' ulang")
                #self.ConnectButton.isChecked = False
                self.ConnectButton.setText("Disconnect")

        else:
            self.serial.disconnect_serial()
            self.BaudrateList.setEnabled(True)
            self.PortList.setEnabled(True)
            self.SaveButton.setEnabled(False)
            self.ProcessButton.setEnabled(False)
            self.ScanButton.setEnabled(False)

            self.ConnectButton.setText("CONNECT")
            self.textDisplay.clear()
            self.textDisplay.append("Device Disconnected")
            self.plot_zero()
            self.RefreshButton.setEnabled(True)
    

    def send_data(self):
        self.textDisplay.clear()

        if(self.serial.serialPort.is_open):
            print("Port Open")
            data_send = 'g'
            self.textDisplay.append("Reading Sensor Data...")
            self.serial.send_data(data_send)
            
        else:
            print("Port Closed")
            self.textDisplay.append("Error, Please Restart Connection")


    def predict_data(self):
        self.textDisplay.clear()
        fig,ax=plt.subplots(figsize=(10,10))
        # ax.imshow(self.data_sensor,cmap="seismic")
        ax.imshow(self.data_sensor,cmap="Reds")
        # ax.imshow(self.data_sensor,cmap="seismic",vmin=0,vmax=4094)
        plt.savefig("temp_footprint_image.png", dpi=300)
        
        image_path = "temp_footprint_image.png"
        
        # # Get image data for prediction
        # image_data = plt.subplots(figsize=(10,10))  # Convert to unsigned 32-bit integers
        # image = image_data.imshow(self.data_sensor,cmap="Reds")
        # image_path = plt.savefig("temp_footprint_image.png")
        # image.save(image_path)
        # image_path = "kaki4.png"
        
        # fig,ax=plt.subplots(figsize=(10,10))
        # image_path = ax.imshow(self.data_sensor)
        
        # prepare image for prediction
        img = keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
        x = keras.preprocessing.image.img_to_array(img)
        x /= 255 
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        
        # Predict
        prediction_Foot = model.predict(images)
        print(prediction_Foot)

        if prediction_Foot >= 0.5:
            self.textDisplay.append(21 * "=")
            self.textDisplay.append("== Telapak Kaki Tidak Normal ==")
            self.textDisplay.append(21 * "=")
            print(33 * "=")
            print("=== Telapak Kaki Tidak Normal ===")
            print(33 * "=")
            
        else:
            self.textDisplay.append(21 * "=")
            self.textDisplay.append("==     Telapak Kaki Normal     ==")
            self.textDisplay.append(21 * "=")
            print(33 * "=")
            print("===    Telapak Kaki Normal    ===")
            print(33 * "=")
    
            
    def update_ports(self):
        self.serial.update_port()
        self.PortList.clear()
        self.PortList.addItems(self.serial.portList)

    def clear(self):
        self.textDisplay.clear()

    #def show(self):
        # command to run
        #self.main_w.show()        

if __name__=='__main__':
    app = QApplication(sys.argv)
    w = MainWidget()
    w.show()
    sys.exit(app.exec_())
        

