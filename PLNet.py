import sys
sys.setrecursionlimit(5000)
from PIL import Image
from rembg.bg import remove
import io
import os
import pandas as pd
import math
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
import reportlab.platypus
from reportlab.lib.styles import getSampleStyleSheet
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QHBoxLayout, QFileDialog, \
     QProgressBar, QComboBox, QMenu, QAction,QScrollArea, QTableWidget, QBoxLayout, QMessageBox,\
     QTableWidgetItem, QListWidget, QDesktopWidget, QLineEdit
from PyQt5.QtGui import QPixmap, QIcon, QFont, QPalette, QColor, QCursor
from PyQt5.QtCore import QSize, Qt, QPoint
import onnxruntime as ort
import numpy as np
from numpy import stack
# self-defined modules
# if getattr(sys, 'frozen', False):
#     app_path = os.path.dirname(sys.executable)
# else:
#     app_path = os.path.dirname(__file__)

# def resource_path(relative_path):
#     """ Get absolute path to resource, works for dev and for PyInstaller """
#     try:
#         # PyInstaller creates a temp folder and stores path in _MEIPASS
#         base_path = sys._MEIPASS
#     except Exception:
#         base_path = os.path.abspath(".")

#     return os.path.join(base_path, relative_path)
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)
# print(app_path)
class predict_model():
    def __init__(self):
        super().__init__()
        self.mymodels = {'DenseNet': resource_path(r'res\\model\\densenet169.onnx'), 'EfficientNet': resource_path(r'res\\model\\model_efficientnet.onnx'), 'ResNet': resource_path(r'res\\model\\resnet50.onnx')}
    def preprocess(self, img):
        # 将PIL图像转换为NumPy数组
        img_array = np.array(img)
        
        # 将像素值缩放到0-1之间
        img_array = img_array.astype(np.float32) / 255.0
        
        # 对像素值进行归一化
        mean = np.array([0.5762883, 0.45526023, 0.32699665], dtype=np.float32)
        std = np.array([0.08670782, 0.09286641, 0.09925108], dtype=np.float32)
        img_array = (img_array - mean) / std
        
        # 将通道维度移至第一个位置
        img_array = np.transpose(img_array, (2, 0, 1))
        
        return img_array
    
    def get_file_size(self, file_path):
        size = os.path.getsize(file_path)
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.2f} KB"
        else:
            return f"{size / (1024 * 1024):.2f} MB"
        
    def crop_image(self, img):

        # Obtain the width and height of the image
        width, height = img.size
        # Create an empty list to store cropped images
        blocks = []
        # Starting from the top left corner of the image, traverse each row and column of the image
        for x in range(width):
            x1 = img.crop((x,0,x+1,height))
            if x1.getextrema() != ((0, 0), (0, 0), (0, 0)):
                w=x
                # print(h)
                break
        for y in range(height):
            x2 = img.crop((0,y,width,y+1))
            if x2.getextrema() != ((0, 0), (0, 0), (0, 0)):                
                h=y
                break
        for x in range(w, width, 224):
            for y in range(h, height, 224):
                block = img.crop((x,y,x+224,y+224))
                blocks.append(block)
        return blocks
 
    def count_black(self, img):
        # Obtain the width and height of the image
        width, height = img.size
        # Create a variable to record the number of black pixels
        black = 0
        # Traverse each pixel of the image
        for x in range(width):
            for y in range(height):
                # If the pixel value is (0,0,0), the number of black pixels is increased by one
                if img.getpixel((x,y)) == (0,0,0):
                    black += 1
        # Calculate the proportion of black pixels to the entire image, retaining two decimal places
        ratio = round(black / (width * height), 2)
        return ratio
   
    def predict_predict(self, fileName, model=None):

        global filename
        filename = None

        if model == None :
            model = self.mymodels['EfficientNet']
            model_path = model
            filename = fileName
            # If you don't know which version is the latest, just choose the default DEFAULT
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        elif model == self.mymodels['EfficientNet']:
            model_path = model
            filename = fileName
            # If you don't know which version is the latest, just choose the default DEFAULT
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        elif model == self.mymodels['DenseNet']:
            model_path = model
            filename = fileName
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        elif model == self.mymodels['ResNet']:
            model_path = model
            filename = fileName
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])


        # Traverse all images in the folder
        byz = 'Corypha umbraculifera'
        tz = 'Borassus flabellifer'
        print(filename)
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.JPG') or filename.endswith('.bmp'):
            # Read the image and cut it into blocks
            img_path = filename
            # with open(img_path,"rb" ) as f:
            #     t = f.read()
            # result_a = remove(data=t)
            # result_Image = Image.open(io.BytesIO(result_a)).convert('RGB')
            result_Image = Image.open(img_path).convert('RGB')

            img = Image.open(img_path).convert('RGB')
            # img = cv2.imread(img_path)
            h, w= result_Image.size
            blocks = self.crop_image(result_Image)
            # blocks = crop_image( img)

            outputbyz = []
            outputtz = []
            # Predict the category of each block and count it

            batch_size = min(64, len(blocks))  # batch size不超过64
            for i in range(0, len(blocks), batch_size):
                batch = blocks[i:i + batch_size]
                batch = stack([self.preprocess(block) for block in batch])
                input_name = session.get_inputs()[0].name
                # 运行模型
                outputs = session.run(None, {input_name: batch})
                # predictions = outputs[0]
                outputbyz.extend(outputs[0][:, 0])
                outputtz.extend(outputs[0][:, 1])
        result = byz if sum(outputbyz)>sum(outputtz) else tz
        a = ['Path:'+img_path+\
        '\nProbability:'+str(round(math.exp(max(sum(outputbyz),sum(outputtz))/len(blocks))/(math.exp(max(sum(outputbyz),\
        sum(outputtz))/len(blocks))+math.exp(min(sum(outputbyz),sum(outputtz))/len(blocks)))*100,2))+'%'+'\nPrediction Result:'+str(result)]
        # print(result_Image.info['dpi'] if 'dpi' in result_Image.info else None)

        return a, h, w, os.path.basename(img_path).split('.')[0], os.path.basename(img_path).split('.')[1], self.get_file_size(img_path), \
            img.info['dpi'] if 'dpi' in img.info else None, str(result)
            # self.labelpredictone.setText('Path:'+img_path+'\nPredict Result:'+str(byz if sum(outputbyz)>sum(outputtz) else tz)+'\nProbability:'+str(round(math.exp(max(sum(outputbyz),sum(outputtz)))/(math.exp(max(sum(outputbyz),sum(outputtz)))+math.exp(min(sum(outputbyz),sum(outputtz))))*100,2))+'%')
    
    def predict_predict_rembg(self, fileName, model = None):
        global filename
        filename = None
        if model is None:
            model = self.mymodels['EfficientNet']
            model_path = model
            filename = fileName
            # If you don't know which version is the latest, just choose the default DEFAULT
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        elif model == self.mymodels['EfficientNet']:
            model_path = model
            filename = fileName
            # If you don't know which version is the latest, just choose the default DEFAULT
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        elif model == self.mymodels['DenseNet']:
            model_path = model
            filename = fileName
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        elif model == self.mymodels['ResNet']:
            model_path = model
            filename = fileName
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])


        # Traverse all images in the folder
        byz = 'Corypha umbraculifera'
        tz = 'Borassus flabellifer'

        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.JPG') or filename.endswith('.bmp'):
            # Read the image and cut it into blocks
            img_path = filename
            with open(img_path,"rb" ) as f:
                t = f.read()
            result_a = remove(data=t)
            result_Image = Image.open(io.BytesIO(result_a)).convert('RGB')

            img = Image.open(img_path).convert('RGB')
            # img = cv2.imread(img_path)
            h, w= result_Image.size
            blocks = self.crop_image_rembg(result_Image)
            # blocks = crop_image( img)

            outputbyz = []
            outputtz = []
            # Predict the category of each block and count it

            batch_size = min(64, len(blocks))  # batch size不超过64
            for i in range(0, len(blocks), batch_size):
                batch = blocks[i:i + batch_size]
                batch = stack([self.preprocess(block) for block in batch])
                input_name = session.get_inputs()[0].name
                # 运行模型
                outputs = session.run(None, {input_name: batch})
                # predictions = outputs[0]
                outputbyz.extend(outputs[0][:, 0])
                outputtz.extend(outputs[0][:, 1])
        result = byz if sum(outputbyz)>sum(outputtz) else tz
        a = ['Path:'+img_path+\
        '\nProbability:'+str(round(math.exp(max(sum(outputbyz),sum(outputtz))/len(blocks))/(math.exp(max(sum(outputbyz),\
        sum(outputtz))/len(blocks))+math.exp(min(sum(outputbyz),sum(outputtz))/len(blocks)))*100,2))+'%'+'\nPrediction Result:'+str(result)]
        # print(result_Image.info['dpi'] if 'dpi' in result_Image.info else None)

        return a, h, w, os.path.basename(img_path).split('.')[0], os.path.basename(img_path).split('.')[1], self.get_file_size(img_path), \
            img.info['dpi'] if 'dpi' in img.info else None, str(result)
            # self.labelpredictone.setText('Path:'+img_path+'\nPredict Result:'+str(byz if sum(outputbyz)>sum(outputtz) else tz)+'\nProbability:'+str(round(math.exp(max(sum(outputbyz),sum(outputtz)))/(math.exp(max(sum(outputbyz),sum(outputtz)))+math.exp(min(sum(outputbyz),sum(outputtz))))*100,2))+'%')

    def predict_all(self, fileName, model = None):
        global filename
        filename = None
        if model is None:
            model = self.mymodels['EfficientNet']
            model_path = model
            filename = fileName
            # If you don't know which version is the latest, just choose the default DEFAULT
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])


        elif model == self.mymodels['EfficientNet']:
            model_path = model
            filename = fileName
            # If you don't know which version is the latest, just choose the default DEFAULT
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        elif model == self.mymodels['DenseNet']:
            model_path = model
            filename = fileName
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        elif model == self.mymodels['ResNet']:
            model_path = model
            filename = fileName
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        classes = ['byz', 'tz']
        counter = {c: 0 for c in classes}
        other = {o: 0 for o in classes}
        byz = 'Corypha umbraculifera'
        tz = 'Borassus flabellifer'

        # self.tableWidget.setRowCount(len(folder_path))
        

        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.JPG') or filename.endswith('.bmp'):
            img_path = filename
            # with open(img_path,"rb" ) as f:
            #     t = f.read()
            # result_a = remove(data=t)
            # result_Image = Image.open(io.BytesIO(result_a)).convert('RGB')
            result_Image = Image.open(img_path).convert('RGB')

            h, w= result_Image.size
            blocks = self.crop_image(result_Image)
            results = []
            outputbyz = []
            outputtz = []
            probbyz = []
            probtz = []
            # total = 3478
            # probbyz = []
            # probtz = []
            batch_size = min(64, len(blocks))  # Batch size does not exceed 64
            for i in range(0, len(blocks), batch_size):
                batch = blocks[i:i + batch_size]
                batch = stack([self.preprocess(block) for block in batch])
                input_name = session.get_inputs()[0].name
                # 运行模型
                outputs = session.run(None, {input_name: batch})
                # predictions = outputs[0]
                outputbyz.extend(outputs[0][:, 0])
                outputtz.extend(outputs[0][:, 1])
            prob_byz = round(math.exp(max(sum(outputbyz),sum(outputtz))/len(blocks))/(math.exp(max(sum(outputbyz),sum(outputtz))/len(blocks))+math.exp(min(sum(outputbyz),sum(outputtz))/len(blocks)))*100,2)
            prob_tz = 100-prob_byz
            df = pd.DataFrame({
            'filename': img_path,
            'prob_byz':[str(prob_byz)+'%'],
            'prob_tz':[str(prob_tz)+'%'],
            'result':[byz if sum(outputbyz)>sum(outputtz) else tz]
            }) 
            return img_path, max(prob_byz,prob_tz), 0 if sum(outputbyz)>sum(outputtz) else 1, h, w
        
    def predict_all_rembg(self, fileName, model = None):
        global filename
        filename = None
        if model is None:
            model = self.mymodels['EfficientNet']
            model_path = model
            filename = fileName
            # If you don't know which version is the latest, just choose the default DEFAULT
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        elif model == self.mymodels['EfficientNet']:
            model_path = model
            filename = fileName
            # If you don't know which version is the latest, just choose the default DEFAULT
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        elif model == self.mymodels['DenseNet']:
            model_path = model
            filename = fileName
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        elif model == self.mymodels['ResNet']:
            model_path = model
            filename = fileName
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        classes = ['byz', 'tz']
        counter = {c: 0 for c in classes}
        other = {o: 0 for o in classes}
        byz = 'Corypha umbraculifera'
        tz = 'Borassus flabellifer'

        # self.tableWidget.setRowCount(len(folder_path))
        
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.JPG') or filename.endswith('.bmp'):
            img_path = filename
            with open(img_path,"rb" ) as f:
                t = f.read()
            result_a = remove(data=t)
            result_Image = Image.open(io.BytesIO(result_a)).convert('RGB')
            h, w= result_Image.size

            blocks = self.crop_image(result_Image)
            results = []
            outputbyz = []
            outputtz = []
            batch_size = min(64, len(blocks))  # batch size不超过64
            for i in range(0, len(blocks), batch_size):
                batch = blocks[i:i + batch_size]
                batch = stack([self.preprocess(block) for block in batch])
                input_name = session.get_inputs()[0].name
                # 运行模型
                outputs = session.run(None, {input_name: batch})
                # predictions = outputs[0]
                outputbyz.extend(outputs[0][:, 0])
                outputtz.extend(outputs[0][:, 1])
            prob_byz = round(math.exp(max(sum(outputbyz),sum(outputtz))/len(blocks))/(math.exp(max(sum(outputbyz),sum(outputtz))/len(blocks))+math.exp(min(sum(outputbyz),sum(outputtz))/len(blocks)))*100,2)
            prob_tz = 100-prob_byz
            df = pd.DataFrame({
            'filename': img_path,
            'prob_byz':[str(prob_byz)+'%'],
            'prob_tz':[str(prob_tz)+'%'],
            'result':[byz if sum(outputbyz)>sum(outputtz) else tz]
            }) 
            
            return img_path, max(prob_byz,prob_tz), 0 if sum(outputbyz)>sum(outputtz) else 1, h, w
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.mymodels = {'DenseNet': resource_path(r'res\\model/densenet169.onnx'), 'EfficientNet': resource_path(r'res\\model\\model_efficientnet.onnx'), 'ResNet': resource_path(r'res\\model\\resnet50.onnx'), 'Your own model': resource_path(r'res\\model\\mymodel.onnx')}
        self.model = resource_path(r'res\\model\\model_efficientnet.onnx')
        self.title = 'Palm Leaf Manuscript Plant Species Identification Software (Mode1)'
        self.left = 400
        self.top = 50
        self.width = 1000
        self.height = 600
        self.menuwidth = 100
        self.menuheight = 30
        self.smallwidth = 120
        self.smallheight = 30
        self.menuoriginal = 30
        self.menucolor = 'black'
        self.background_color = 'white'
        self.button_color = 'white'
        self.setWindowIcon(QIcon(resource_path(r'res\\photo\\1694874874141.png')))
        self.helpfile = resource_path(r'\\res\\User_manuals.pdf')
        self.introduceCorypha = resource_path(r'res\\introduceCorypha.pdf')
        self.introduceBorassus = resource_path(r'res\\introduceBorassus.pdf')
        # self.imageInfo = []
        self.initUI()
        # Initialize resolution and color options
        self.resolution_actions = []
        self.color_actions = []
        self.arb_actions = []
        self.model_actions = []
        self.init_setmenus()
        self.predict = None
        self.predict2 = None
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setMaximumSize(self.width, self.height) 
        self.resize(self.width, self.height)
        # Window centered
        self.center()

        self.imageInfo = []
        # self.scroArea = QScrollArea(self)
        # layout = QBoxLayout()
        topLayout = QHBoxLayout(self)
        # Create a button named "File"
        self.file_button = QPushButton("File", self)
        self.file_button.setGeometry(0, 0, self.menuwidth, self.menuheight)
        self.file_button.setMenu(None)
        self.file_button.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                color: black;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: rgb(240, 240, 240); 
            }
            QPushButton::menu-indicator {
                image: none;
            }
            """
        )
        # Create a menu for the "File" button
        self.menu = QMenu(self)
        self.file_button.setMenu(self.menu)
        # Create two menu items: "100" and "1000"
        self.item_100 = QAction("Open File...", self)
        self.item_1000 = QAction("Open Folder...", self)
        # Add the menu items to the menu
        self.menu.addAction(self.item_100)
        self.menu.addAction(self.item_1000)
        self.file_button.setHidden(False)

        self.file_button2 = QPushButton("File", self)
        self.file_button2.setGeometry(0, 0, self.menuwidth, self.menuheight)
        self.file_button2.setMenu(None)
        self.file_button2.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                color: black;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: rgb(240, 240, 240); 
            }
            QPushButton::menu-indicator {
                image: none;
            }
            """
        )
        # Create a menu for the "File" button
        self.menu2 = QMenu(self)
        self.file_button2.setMenu(self.menu2)
        # Create two menu items: "100" and "1000"
        self.item_100_2 = QAction("Open File...", self)
        self.item_1000_2 = QAction("Open Folder...", self)
        # Add the menu items to the menu
        self.menu2.addAction(self.item_100_2)
        self.menu2.addAction(self.item_1000_2)
        self.file_button2.setHidden(True)

        # Create "Settings" button
        self.settings_button = QPushButton('Setting', self)
        self.settings_button.setToolTip('Click to change settings')
        self.settings_button.move(2*self.menuwidth, 0)
        self.settings_button.resize(self.menuwidth, self.menuheight)
        self.settings_button.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                color: black;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: rgb(240, 240, 240); /* Darker blue on hover */
            }
            QPushButton::menu-indicator {
                image: none;
            }
            """
        )
        self.settings_button.clicked.connect(self.open_setmenu)

        # Create "Tools" button
        self.tools_button = QPushButton('Tool', self)
        self.tools_button.setToolTip('Click to choose tools')
        self.tools_button.move(3*self.menuwidth, 0)
        self.tools_button.resize(self.menuwidth, self.menuheight)
        self.tools_button.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                color: black;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: rgb(240, 240, 240); /* Darker blue on hover */
            }
            QPushButton::menu-indicator {
                image: none;
            }
            """
        )
        self.tools_button.clicked.connect(self.open_toolmenu)

        self.button_help = QPushButton('Help', self)
        self.button_help.setToolTip('Click to get help')
        self.button_help.move(4*self.menuwidth, 0)
        self.button_help.resize(self.menuwidth, self.menuheight)
        self.button_help.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                color: black;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: rgb(240, 240, 240); /* Darker blue on hover */
            }
            QPushButton::menu-indicator {
                image: none;
            }
            """
        )
        self.button_help.clicked.connect(self.open_pdf)

        self.button_mode = QPushButton('Mode', self)
        self.button_mode.setToolTip('Click to change Mode')
        self.button_mode.move(1*self.menuwidth, 0)
        self.button_mode.resize(self.menuwidth, self.menuheight)
        self.button_mode.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                color: black;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: rgb(240, 240, 240); /* Darker blue on hover */
            }
            QPushButton::menu-indicator {
                image: none;
            }
            """
        )
        self.button_mode.clicked.connect(self.toggleLabel)
        self.button_mode.clicked.connect(self.changeTitle)

        self.button = QPushButton('', self)
        self.button.setToolTip('Click to select an image')
        self.button_icon = QIcon(resource_path("res\\photo\\transparent_icon_file3.png"))
        self.button.setIcon(self.button_icon)
        self.button.setIconSize(QSize(60, 60))
        self.button.move(0*60, self.menuoriginal)
        self.button.resize(60, 60)
        self.button.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                color: black;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: rgb(240, 240, 240); 
            }
            QPushButton::menu-indicator {
                image: none;
            }
            """
        )
        self.item_100.triggered.connect(self.loadoneImage)
        # self.button.clicked.connect(self.on_button_click)
        self.button.clicked.connect(self.loadoneImage)

        self.button2 = QPushButton('', self)
        self.button2.setToolTip('Click to select an image')
        self.button2_icon = QIcon(resource_path("res\\photo\\transparent_icon_file3.png"))
        self.button2.setIcon(self.button2_icon)
        self.button2.setIconSize(QSize(60, 60))
        self.button2.move(0*60, self.menuoriginal)
        self.button2.resize(60, 60)
        self.button2.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                color: black;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: rgb(240, 240, 240); 
            }
            QPushButton::menu-indicator {
                image: none;
            }
            """
        )
        self.item_100_2.triggered.connect(self.generateoneDataFrame)
        self.button2.clicked.connect(self.generateoneDataFrame)
        self.button2.setHidden(True)

        self.button_predict1 = QPushButton('', self)
        self.button_predict1.setToolTip('Click to select a folder')
        self.button_predict1_icon = QIcon(resource_path("res\\photo\\transparent_icon_folder3.png"))
        self.button_predict1.setIcon(self.button_predict1_icon)
        self.button_predict1.setIconSize(QSize(60, 60))
        self.button_predict1.move(1*60, self.menuoriginal)
        self.button_predict1.resize(60, 60)
        self.button_predict1.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                color: black;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: rgb(240, 240, 240); /* Darker blue on hover */
            }
            QPushButton::menu-indicator {
                image: none;
            }
            """
        )
        self.item_1000.triggered.connect(self.loadImages)
        self.button_predict1.setHidden(False)
        self.button_predict1.clicked.connect(self.loadImages)

        self.button_predict2 = QPushButton('', self)
        self.button_predict2.setToolTip('Click to select a folder')
        self.button_predict2_icon = QIcon(resource_path("res\\photo\\transparent_icon_folder3.png"))
        self.button_predict2.setIcon(self.button_predict2_icon)
        self.button_predict2.setIconSize(QSize(60, 60))
        self.button_predict2.move(1*60, self.menuoriginal)
        self.button_predict2.resize(60, 60)
        self.button_predict2.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                color: black;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: rgb(240, 240, 240); /* Darker blue on hover */
            }
            QPushButton::menu-indicator {
                image: none;
            }
            """
        )
        self.item_1000_2.triggered.connect(self.generateDataFrame)
        self.button_predict2.clicked.connect(self.generateDataFrame)
        self.button_predict2.setHidden(True)

        self.show_password_button = QPushButton('', self)
        self.show_password_button.setToolTip('Click to Save the table')
        self.show_password_button_icon = QIcon(resource_path("res\\photo\\transparent_icon_save3.png"))
        self.show_password_button.setIcon(self.show_password_button_icon)
        self.show_password_button.setIconSize(QSize(60, 60))
        self.show_password_button.move(2*60, self.menuoriginal)
        self.show_password_button.resize(60, 60)
        self.show_password_button.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                color: black;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: rgb(240, 240, 240); 
            }
            QPushButton::menu-indicator {
                image: none;
            }
            """
        )
        self.show_password_button.clicked.connect(self.saveTable)

        self.button_delete = QPushButton('', self)
        self.button_delete.setToolTip('Click to delete the selected line')
        self.button_delete_icon = QIcon(resource_path("res\\photo\\transparent_icon_delete3.png"))
        self.button_delete.setIcon(self.button_delete_icon)
        self.button_delete.setIconSize(QSize(60, 60))

        self.button_delete.setGeometry(3*60, self.menuoriginal, 60, 60)
        self.button_delete.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                color: black;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: rgb(240, 240, 240); /* Darker blue on hover */
            }
            QPushButton::menu-indicator {
                image: none;
            }   
            """
        )
        self.button_delete.clicked.connect(self.deleteSelected)
        self.button_delete.setHidden(False)

        self.button_clear0 = QPushButton('', self)
        self.button_clear0.setToolTip('Click to clear the result')
        self.button_clear0_icon = QIcon(resource_path("res\\photo\\transparent_clear3.png"))
        self.button_clear0.setIcon(self.button_clear0_icon)
        self.button_clear0.setIconSize(QSize(60, 60))
        self.button_clear0.setGeometry(4*60, self.menuoriginal, 60, 60) 
        self.button_clear0.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                color: black;   
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: rgb(240, 240, 240); /* Darker blue on hover */
            }
            QPushButton::menu-indicator {
                image: none;
            }
            """
        )
        self.button_clear0.clicked.connect(self.on_button_clear_click)

        # Generate PDF report
        self.button_pdf1 = QPushButton('', self)
        self.button_pdf1.setToolTip('Click to generate the pdf report')
        self.button_pdf1_icon = QIcon(resource_path("res\\photo\\report2.png"))
        self.button_pdf1.setIcon(self.button_pdf1_icon)
        self.button_pdf1.setIconSize(QSize(60, 60))
        self.button_pdf1.setGeometry(5*60, self.menuoriginal, 60, 60) 
        self.button_pdf1.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                color: black;   
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: rgb(240, 240, 240); /* Darker blue on hover */
            }
            QPushButton::menu-indicator {
                image: none;
            }
            """
        )
        self.button_pdf1.clicked.connect(self.savepdf1)

        # Generate PDF report
        self.button_pdf2 = QPushButton('', self)
        self.button_pdf2.setToolTip('Click to generate the pdf report')
        self.button_pdf2_icon = QIcon(resource_path("res\\photo\\report2.png"))
        self.button_pdf2.setIcon(self.button_pdf2_icon)
        self.button_pdf2.setIconSize(QSize(60, 60))
        self.button_pdf2.setGeometry(3*60, self.menuoriginal, 60, 60) 
        self.button_pdf2.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                color: black;   
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: rgb(240, 240, 240); /* Darker blue on hover */
            }
            QPushButton::menu-indicator {
                image: none;
            }
            """
        )
        self.button_pdf2.setHidden(True)
        self.button_pdf2.clicked.connect(self.savepdf2)

        # self.close_button = QPushButton('Close',self)
        # self.close_button.setToolTip('Click to close the window')
        # self.close_button.setGeometry(350, 550, 100, 30)
        # self.close_button.clicked.connect(self.on_close_button_click)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(100, 95, 800, self.smallheight)
        self.progress_bar.setStyleSheet("background-color: {};color: {};border: 1px solid black;".format(self.background_color, self.menucolor))
        self.progress_bar.setHidden(True)

        self.label_progressbar = QLabel(self)
        self.label_progressbar.move(20, 95)
        self.label_progressbar.resize(60, self.smallheight)
        # label_progressbar.setStyleSheet("background-color: {};color: {};border: 1px solid black;".format(self.background_color, self.menucolor))    
        self.label_progressbar.setText('Progress:')
        self.label_progressbar.setHidden(True)

        self.image = None
        self.zoom_factor = 1.0

        self.label = QLabel(self)
        self.label.setGeometry(100, 130, 800, 300)
        self.label.setStyleSheet("background-color: {};color: {};border: 1px solid black;".format(self.background_color, self.menucolor))

        self.imageLabel = QLabel(self)
        self.imageLabel.setGeometry(300, 90, 700, 400)
        self.imageLabel.setStyleSheet("background-color: {};color: {};border: 1px solid black;".format(self.background_color, self.menucolor))
        # self.imageLabel.setAlignment(Qt.AlignCenter)
        # self.imageLabel.setFixedSize(700, 400)
        self.imageLabel.setHidden(False)


        self.label_menu = QLabel(self)
        self.label_menu.move(0, 0)
        self.label_menu.resize(self.width, self.smallheight)
        self.label_menu.setStyleSheet(
            """
                background-color: transparent;
                color: black;
                padding: 5px 10px;
                border-top: 1px solid gray; 
                border-bottom: 1px solid gray;
            """
        )
        self.label_menu.lower() 

        self.label_menu2 = QLabel(self)
        self.label_menu2.move(0, 30)
        self.label_menu2.resize(self.width, 60)
        self.label_menu2.setStyleSheet(
            """
                background-color: transparent;
                color: black;
                padding: 5px 10px;
                border-bottom: 1px solid gray;
            """
        )
        self.label_menu2.lower()

        #Create a QFont object and set the font size
        self.font = QFont()
        self.font.setPointSize(15)  # Set the font size to 15
        # self.font.setBold(True)  # Set to bold
        # self.font.setUnderline(True)  # Set to underline

        self.labelpredictone = QLabel(self)
        self.labelpredictone.setGeometry(300, 490, 700, 110)
        self.labelpredictone.setStyleSheet("background-color: {};color: {};border: 1px solid black;".format(self.background_color, self.menucolor))
        self.labelpredictone.setFont(self.font)
        self.palette = self.labelpredictone.palette()  # Get the palette
        self.palette.setColor(QPalette.WindowText, QColor('blue'))  # Set the font color to blue
        self.labelpredictone.setPalette(self.palette)
        #set the text to be selectable
        self.labelpredictone.setTextInteractionFlags(Qt.TextSelectableByMouse)
        #set the cursor to be I-beam cursor
        self.labelpredictone.setCursor(QCursor(Qt.IBeamCursor ))

        self.label_for = QLabel(self)
        self.label_for.move(500, 500)
        self.label_for.resize(100, 20)

        self.label_form = QLabel(self)
        self.label_form.setGeometry(100, 130, 800, 400)
        self.label_form.setStyleSheet("background-color: {};color: {};border: 1px solid black;".format(self.background_color, self.menucolor))
        self.label_form.setHidden(True)

        self.table_widget = QTableWidget(self)
        self.table_widget.setGeometry(100, 130, 800, 400)
        self.table_widget.setColumnCount(3)
        self.table_widget.setHorizontalHeaderLabels(['Path', 'Probablity', 'Result'])
        self.table_widget.setColumnWidth(0, 350)  # Set width of column  to 350
        self.table_widget.setColumnWidth(1, 100) 
        self.table_widget.setColumnWidth(2, 320)
        self.table_widget.setHidden(True)

        self.labelrusultform = QLabel(self)
        self.labelrusultform.setGeometry(20, 130, 60, 30)   
        self.labelrusultform.setAlignment(Qt.AlignCenter)
        self.labelrusultform.setText('Result:')
        self.labelrusultform.setHidden(True)

        #label path
        self.label_path_bar = QLabel(self)
        self.label_path_bar.setGeometry(0, 90, 300, 30)   
        self.label_path_bar.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.label_path_bar.setText('File Informatiom')
        self.label_path_bar.setStyleSheet("border: 1px solid black;")
        self.label_path_bar.setHidden(False)

        self.listWidget = QListWidget(self)
        self.listWidget.setGeometry(0, 120, 300, 480)   
        self.listWidget.setStyleSheet("background-color: {};color: {};border: 1px solid black;".format(self.background_color, self.menucolor))
        self.listWidget.setHidden(False)
        self.listWidget.clicked.connect(self.showSelectedImage)

    def deleteSelected(self):
        selected_item = self.listWidget.currentItem()
        if selected_item:
            row = self.listWidget.row(selected_item)
            self.listWidget.takeItem(row)
            self.imageLabel.clear()
            self.labelpredictone.clear()

    def loadoneImage(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files(*);;*.bmp;;*.jpg;;*.png;;*.JPG;;*.jpeg", options=options)
        if fileName:
            self.listWidget.addItem(os.path.normpath(fileName))

    def loadImages(self):
        folder_path = QFileDialog.getExistingDirectory(self, "choose file")
        if folder_path:
            # self.listWidget.clear()
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    self.listWidget.addItem(os.path.normpath(os.path.join(folder_path, file_name)))

    def showSelectedImage(self):
        
        selected_item = self.listWidget.currentItem()
        if selected_item:
            pixmap = QPixmap(selected_item.text())
            # print(selected_item.text())
            # print(type(selected_item.text()))
            a, h, w, imagename, imageformat, filesize, imagedpi, imagepredict  = self.predict_predict(fileName = selected_item.text())
            
            self.imageInfo = [a, h, w, imagename, imageformat, filesize, imagedpi, imagepredict]
            self.labelpredictone.setText(a[0])

            # Adjust image size to fit 500x500 labels
            # self.scroll_area.setPixmap(pixmap.scaled(700, 400, Qt.KeepAspectRatio))
            self.imageLabel.setPixmap(pixmap.scaled(700, 400, Qt.KeepAspectRatio))
    
    def savepdf1(self):
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Report", "", "PDF Files (*.pdf)")
        if save_path:
            # Method of displaying Chinese
            # pdfmetrics.registerFont(TTFont('SimSun', 'simsun.ttf'))

            doc = SimpleDocTemplate(save_path, pagesize=letter)
            elements = []
            styles = getSampleStyleSheet()

            # styles['Title'].fontName = 'SimSun'
            title = 'Imformation Statistics of Palm Leaf Manuscripts'
            elements.append(Paragraph(title, styles['Title']))

            data = [
                ['Image Name',self.imageInfo[3],'Image format',self.imageInfo[4]],
                ['File Size',self.imageInfo[5],'Image Size',f'{self.imageInfo[1]} x {self.imageInfo[2]}'],
                ['Image DPI',str(self.imageInfo[6]),'Plant Genera',self.imageInfo[7]],
                ['Image',reportlab.platypus.Image(self.listWidget.currentItem().text(), 390, 90)]
            ]

            # Create the table with the data
            t = Table(data, colWidths=[100, 150, 100, 150], rowHeights=[30,30,30,100])

            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.white),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (2, 0), (2, 2), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                # ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('SPAN', (1, 3), (3, 3)),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),

            ]))

            # Add the table to the elements list
            elements.append(t)
            elements.append(Spacer(2, 10))  # Add 20 units of space below the table
            note = Paragraph("Note:The contents are provided for your information only and are provided on an 'as is' and 'as available' basis.", styles["Normal"])  # Create a note paragraph
            elements.append(note)  # Add notes to the element list
            # Build the PDF document with the elements
            doc.build(elements)
            QMessageBox.information(self, 'Report Generated', 'The report has been generated successfully.')

    def savepdf2(self):

        save_path, _ = QFileDialog.getSaveFileName(self, "Save Report", "", "PDF Files (*.pdf)")
        if save_path:

            # Method of displaying Chinese
            # pdfmetrics.registerFont(TTFont('SimSun', 'simsun.ttf'))

            doc = SimpleDocTemplate(save_path, pagesize=letter)
            elements = []
            styles = getSampleStyleSheet()

            # styles['Title'].fontName = 'SimSun'
            title = 'Imformation Statistics of Palm Leaf Manuscripts'
            elements.append(Paragraph(title, styles['Title']))

            # Create the table data
            # print('xinxi',self.imageInfo)
           

            data = [
                ['File Path',self.imageInfo[0],'',''],
                ['Total files',self.imageInfo[1],'Image Num',self.imageInfo[2]],
                [ 'Borassus Count', f'{self.imageInfo[3]}({self.imageInfo[4]}%)', 'Corypha Count', f'{self.imageInfo[5]}({self.imageInfo[6]}%)'],
                ['probability>95%', f'{self.imageInfo[7]}({self.imageInfo[8]}%)', 'probability<95%', f'{self.imageInfo[9]}({self.imageInfo[10]}%)'],
                ['normal resolution', f'{self.imageInfo[11]}({self.imageInfo[12]}%)', 'poor resolution', f'{self.imageInfo[13]}({self.imageInfo[14]}%)'],
            ]

            # Create the table with the data
            t = Table(data, colWidths=[100, 150, 100, 150], rowHeights=[30,30,30,30,30])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.white),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (2, 1), (2, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                # ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('SPAN', (1, 0), (3, 0)),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))

            # Add the table to the elements list
            elements.append(t)
            elements.append(Spacer(2, 10))  # 在表格下方添加20单位的空间
            note = Paragraph("Note:The Contents are provided for your information only and are provided on an 'as is' and 'as available' basis.", styles["Normal"])  # 创建备注段落
            elements.append(note)  # Add notes to the element list
            # Build the PDF document with the elements
            doc.build(elements)
            QMessageBox.information(self, 'Report Generated', 'The report has been generated successfully.')
    
    def toggleLabel(self):
        if self.label.isHidden():

            self.label.setHidden(False)
            # self.scroll_area.setHidden(False)
            # self.label_result.setHidden(False)
            # self.label_iamge.setHidden(False)
            self.labelpredictone.setHidden(False)
            # self.label.setHidden(False)
            self.label_path_bar.setHidden(False)
            self.listWidget.setHidden(False)
            self.imageLabel.setHidden(False)
            self.button_predict1.setHidden(False)
            self.button_pdf1.setHidden(False)
            self.file_button.setHidden(False)
            self.button.setHidden(False)
            self.button_delete.setHidden(False)

            self.labelrusultform.setHidden(True)
            self.table_widget.setHidden(True)
            # self.form_scroll_area.setHidden(True)
            self.label_progressbar.setHidden(True)
            self.progress_bar.setHidden(True)
            self.label_form.setHidden(True)
            self.button_pdf2.setHidden(True)
            self.button_predict2.setHidden(True)
            self.file_button2.setHidden(True)
            self.button2.setHidden(True)


        else:
            self.label.setHidden(True)
            self.label_form.setHidden(False)
            # self.scroll_area.setHidden(True)
            # self.label_result.setHidden(True)
            # self.label_iamge.setHidden(True)
            self.labelpredictone.setHidden(True)
            self.labelrusultform.setHidden(False)
            self.table_widget.setHidden(False)
            # self.form_scroll_area.setHidden(False)
            self.label_progressbar.setHidden(False)
            self.progress_bar.setHidden(False)
            self.label_path_bar.setHidden(True)
            self.listWidget.setHidden(True)
            self.imageLabel.setHidden(True)
            self.button_predict2.setHidden(False)
            self.button_predict1.setHidden(True)
            self.button_pdf1.setHidden(True)
            self.button_pdf2.setHidden(False)
            self.file_button.setHidden(True)
            self.file_button2.setHidden(False)
            self.button.setHidden(True)
            self.button2.setHidden(False)
            self.button_delete.setHidden(True)

    def center(self):
            # Obtain the frame where the main window is located
            qr = self.frameGeometry()
            # Obtain the resolution of the monitor and then obtain the position of the midpoint
            cp = QDesktopWidget().availableGeometry().center()
            # Set the center point of the window
            qr.moveCenter(cp)
            # Move the top left corner of the main window to the top left corner of its frame using the move function, so that the window is centered
            self.move(qr.topLeft())
    
    def open_pdf(self):
        os.startfile(self.helpfile)

    def on_close_button_click(self):
        sys.exit()

    def on_button_clear_click(self):
        self.label.clear()
        self.labelpredictone.clear()
        self.label_for.clear()
        self.progress_bar.setValue(0)
        # self.label_path_bar.clear()
        self.listWidget.clear()
        self.imageLabel.clear()



    def generateoneDataFrame(self):
        self.imageInfo = []
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files(*);;*.bmp;;*.jpg;;*.png;;*.JPG;;*.jpeg", options=options)
        # fd, type = QFileDialog.getSaveFileName(self, "Please choose a path to save predict result", "", "*.csv")
        if filename:
            self.table_widget.setRowCount(1)
            total = 1
            image_num = 0
            prob95 = 0
            num_b=0
            num_c=0
            pixel256=0


            if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.JPG') or filename.endswith('.bmp') or filename.endswith('.jpeg'):
                a = self.predict_all(fileName = filename)
                # print(a[0],a[1],a[2])
                
                self.table_widget.setItem(0, 0, QTableWidgetItem(a[0]))
                self.table_widget.setItem(0, 1, QTableWidgetItem(str(a[1])+ '%'))
                self.table_widget.setItem(0, 2, QTableWidgetItem('Corypha umbraculifera' if a[2] == 0 else 'Borassus flabellifer'))
                for i in range(self.table_widget.columnCount()):
                    for j in range(self.table_widget.rowCount()):
                        self.table_widget.item(j, i).setTextAlignment(Qt.AlignCenter)
                if a[2] == 1:
                    num_b+=1
                else:
                    num_c+=1
                if a[1] > 95:
                    prob95+=1
                if a[3] >= 256 and a[4] >= 256:
                    pixel256+=1
                self.progress_bar.setValue(int(100*(0 + 1)/1))
                image_num += 1
            self.imageInfo.append(os.path.split(filename)[0])
            self.imageInfo.extend([total, image_num, num_b, round(num_b/image_num*100,2), num_c, round(num_c/image_num*100,2), prob95, 
                                    round(prob95/image_num*100,2), image_num-prob95, round(100-prob95/image_num*100,2), pixel256, round(pixel256/image_num*100,2),
                                    image_num-pixel256, round(100-pixel256/image_num*100,2)])
    
    def generateDataFrame(self):
        self.imageInfo = []
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName = QFileDialog.getExistingDirectory(self,"Please select a folder containing images", "", options=options)
        if fileName:
            # fd, type = QFileDialog.getSaveFileName(self, "Please choose a path to save predict result", "", "*.csv")
            t_RowCount=0
            self.table_widget.setRowCount(t_RowCount)
            total = len(os.listdir(fileName))
            image_num = 0
            prob95 = 0
            num_b=0
            num_c=0
            pixel256=0
            for i,filename in enumerate(os.listdir(fileName)):
                if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.JPG') or filename.endswith('.bmp') or filename.endswith('.jpeg'):
                    img_path = os.path.join(fileName, filename)
                    a = self.predict_all(fileName = img_path)
                    # print(a[0],a[1],a[2])
                    t_RowCount+=1
                    self.table_widget.setRowCount(t_RowCount)
                    self.table_widget.setItem(image_num, 0, QTableWidgetItem(os.path.basename(a[0])))
                    self.table_widget.setItem(image_num, 1, QTableWidgetItem(str(a[1])+ '%'))
                    self.table_widget.setItem(image_num, 2, QTableWidgetItem('Corypha umbraculifera' if a[2] == 0 else 'Borassus flabellifer'))

                    if a[2] == 1:
                        num_b+=1
                    else:
                        num_c+=1
                    if a[1] > 95:
                        prob95+=1
                    if a[3] >= 256 and a[4] >= 256:
                        pixel256+=1
                    self.progress_bar.setValue(int(100*(i + 1)/len(os.listdir(fileName))))
                    image_num += 1

            self.imageInfo.append(os.path.basename(fileName))
            self.imageInfo.extend([total, image_num, num_b, round(num_b/image_num*100,2), num_c, round(num_c/image_num*100,2), prob95, 
                                    round(prob95/image_num*100,2), image_num-prob95, round(100-prob95/image_num*100,2), pixel256, round(pixel256/image_num*100,2),
                                    image_num-pixel256, round(100-pixel256/image_num*100,2)])

            # print(self.imageInfo[0])
            for i in range(self.table_widget.columnCount()):
                for j in range(self.table_widget.rowCount()):
                    if self.table_widget.item(j, i) is None:
                        continue
                    self.table_widget.item(j, i).setTextAlignment(Qt.AlignCenter)
            QMessageBox.information(self, 'Table Generated', 'The table has been generated successfully.')

    def updateTable(self, df):
        self.table_widget.setRowCount(len(df))
        for row_index, row_data in df.iterrows():
            for col_index, value in enumerate(row_data):
                item = QTableWidgetItem(str(value))
                # if row_index == 0:
                #     # Add a lower border for the first row
                #     item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                #     item.setCheckState(Qt.Checked)
                self.table_widget.setItem(row_index, col_index, item)
    
    def saveTable(self):
        folder_path, _ = QFileDialog.getSaveFileName(self, "Please choose a path to save predict result", "", "*.csv")
        
        if folder_path:
            df_list = []
            
            for row in range(self.table_widget.rowCount()):
                if self.table_widget.item(row, 0) is None:
                    continue
                
                i1 = self.table_widget.item(row, 0).text()  # filename
                i2 = self.table_widget.item(row, 1).text()  # probability
                i3 = self.table_widget.item(row, 2).text()  # result
                
                # 计算香农熵
                try:
                    # 处理百分比格式，去掉%符号并转换为小数
                    if isinstance(i2, str) and i2.endswith('%'):
                        p = float(i2.rstrip('%')) / 100.0
                    else:
                        p = float(i2)
                    
                    # 处理边界情况，避免log(0)
                    if p <= 0:
                        entropy = 0
                    elif p >= 1:
                        entropy = 0
                    else:
                        # 香农熵公式: H = -p*log2(p) - (1-p)*log2(1-p)
                        entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
                    
                except (ValueError, TypeError):
                    # 如果概率值无法转换为数字，设置熵为NaN
                    entropy = np.nan
                
                df_list.append({
                    'filename': i1, 
                    'probability': i2,  # 修正拼写错误
                    'result': i3,
                    'shannon_entropy': entropy
                })
            
            df = pd.DataFrame(df_list)
            df.to_csv(folder_path, index=False)
            # self.label_for.setText('Save Done!')  # 取消注释如果需要显示保存完成信息    
    def changeTitle(self):
        if self.windowTitle() == 'Palm Leaf Manuscript Plant Species Identification Software (Mode1)':
            self.setWindowTitle('Palm Leaf Manuscript Plant Species Identification Software (Mode2)')
        else:
            self.setWindowTitle('Palm Leaf Manuscript Plant Species Identification Software (Mode1)')

    def update_image(self):
        if self.image:
            scaled_image = self.image.scaled(int(600 * self.zoom_factor), int(300 * self.zoom_factor), Qt.KeepAspectRatio)
            self.label.setPixmap(QPixmap.fromImage(scaled_image))
    
    def wheelEvent(self, event):
        if self.image:
            num_degrees = event.angleDelta().y() / 8
            num_steps = num_degrees / 15
            self.zoom_factor += num_steps / 10
            self.zoom_factor = max(0.1, min(2.0, self.zoom_factor))
            self.update_image()
    
    def closeEvent(self, event):
        # Ask if you want to close the window
        reply = QMessageBox.question(self, 'Confirm Exit',
                                     "Are you sure you want to exit?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
    
    def crop_image_rembg(self, img):

        # Obtain the width and height of the image
        width, height = img.size
        # Create an empty list to store cropped images
        blocks = []
        # Starting from the top left corner of the image, traverse each row and column of the image
        for x in range(width):
            x1 = img.crop((x,0,x+1,height))
            if x1.getextrema() != ((0, 0), (0, 0), (0, 0)):
                w=x
                # print(h)
                break
        for y in range(height):
            x2 = img.crop((0,y,width,y+1))
            if x2.getextrema() != ((0, 0), (0, 0), (0, 0)):                
                h=y
                break
        for x in range(w, width, 224):
            for y in range(h, height, 224):
                block = img.crop((x,y,x+224,y+224))
                blocks.append(block)
        return blocks
    
    def crop_image(self, img):
        # Use Image to read images and convert them to RGB images
        # img = Image.open(image).convert('RGB')
        # Obtain the width and height of the image
        width, height = img.size
        # Create an empty list to store cropped images
        blocks = []
        # Starting from the top left corner of the image, traverse each row and column of the image
        for x in range(0, width, 224):
            for y in range(0, height, 224):
                block = img.crop((x,y,x+224,y+224))
                blocks.append(block)
        # Finally, output this list
        return blocks

    def get_file_size(self, file_path):
        size = os.path.getsize(file_path)
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.2f} KB"
        else:
            return f"{size / (1024 * 1024):.2f} MB"

    def count_black(self, img):
        # Obtain the width and height of the image
        width, height = img.size
        # Create a variable to record the number of black pixels
        black = 0
        # Traverse each pixel of the image
        for x in range(width):
            for y in range(height):
                # If the pixel value is (0,0,0), the number of black pixels is increased by one
                if img.getpixel((x,y)) == (0,0,0):
                    black += 1
        # Calculate the proportion of black pixels to the entire image, retaining two decimal places
        ratio = round(black / (width * height), 2)
        return ratio
    def preprocess(self, img):
        # 将PIL图像转换为NumPy数组
        img_array = np.array(img)
        
        # 将像素值缩放到0-1之间
        img_array = img_array.astype(np.float32) / 255.0
        
        # 对像素值进行归一化
        mean = np.array([0.5762883, 0.45526023, 0.32699665], dtype=np.float32)
        std = np.array([0.08670782, 0.09286641, 0.09925108], dtype=np.float32)
        img_array = (img_array - mean) / std
        
        # 将通道维度移至第一个位置
        img_array = np.transpose(img_array, (2, 0, 1))
        
        return img_array
    
    def predict_predict(self, fileName):
        model_path = self.model
        filename = fileName
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        # Traverse all images in the folder
        byz = 'Corypha umbraculifera'
        tz = 'Borassus flabellifer'

        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.JPG') or filename.endswith('.bmp'):
            # Read the image and cut it into blocks
            img_path = filename
            # with open(img_path,"rb" ) as f:
            #     t = f.read()
            # result_a = remove(data=t)
            # result_Image = Image.open(io.BytesIO(result_a)).convert('RGB')
            result_Image = Image.open(img_path).convert('RGB')

            img = Image.open(img_path).convert('RGB')
            # img = cv2.imread(img_path)
            h, w= result_Image.size
            blocks = self.crop_image(result_Image)
            # blocks = crop_image( img)

            outputbyz = []
            outputtz = []
            # Predict the category of each block and count it

            batch_size = min(64, len(blocks))  # batch size不超过64
            for i in range(0, len(blocks), batch_size):
                batch = blocks[i:i + batch_size]
                batch = stack([self.preprocess(block) for block in batch])
                input_name = session.get_inputs()[0].name
                # 运行模型
                outputs = session.run(None, {input_name: batch})
                # predictions = outputs[0]
                outputbyz.extend(outputs[0][:, 0])
                outputtz.extend(outputs[0][:, 1])

        result = byz if sum(outputbyz)>sum(outputtz) else tz
        a = ['Path:'+img_path+\
        '\nProbability:'+str(round(math.exp(max(sum(outputbyz),sum(outputtz))/len(blocks))/(math.exp(max(sum(outputbyz),\
        sum(outputtz))/len(blocks))+math.exp(min(sum(outputbyz),sum(outputtz))/len(blocks)))*100,2))+'%'+'\nPrediction Result:'+str(result)]
        # print(result_Image.info['dpi'] if 'dpi' in result_Image.info else None)

        return a, h, w, os.path.basename(img_path).split('.')[0], os.path.basename(img_path).split('.')[1], self.get_file_size(img_path), \
            img.info['dpi'] if 'dpi' in img.info else None, str(result)
            # self.labelpredictone.setText('Path:'+img_path+'\nPredict Result:'+str(byz if sum(outputbyz)>sum(outputtz) else tz)+'\nProbability:'+str(round(math.exp(max(sum(outputbyz),sum(outputtz)))/(math.exp(max(sum(outputbyz),sum(outputtz)))+math.exp(min(sum(outputbyz),sum(outputtz))))*100,2))+'%')
    
    def predict_predict_rembg(self, fileName):
        model_path = self.model
        filename = fileName
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        # Traverse all images in the folder
        byz = 'Corypha umbraculifera'
        tz = 'Borassus flabellifer'

        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.JPG') or filename.endswith('.bmp'):
            # Read the image and cut it into blocks
            img_path = filename
            with open(img_path,"rb" ) as f:
                t = f.read()
            result_a = remove(data=t)
            result_Image = Image.open(io.BytesIO(result_a)).convert('RGB')

            img = Image.open(img_path).convert('RGB')
            # img = cv2.imread(img_path)
            h, w= result_Image.size
            blocks = self.crop_image_rembg(result_Image)
            # blocks = crop_image( img)

            outputbyz = []
            outputtz = []
            # Predict the category of each block and count it

            batch_size = min(64, len(blocks))  # batch size不超过64
            for i in range(0, len(blocks), batch_size):
                batch = blocks[i:i + batch_size]
                batch = stack([self.preprocess(block) for block in batch])
                input_name = session.get_inputs()[0].name
                # 运行模型
                outputs = session.run(None, {input_name: batch})
                # predictions = outputs[0]
                outputbyz.extend(outputs[0][:, 0])
                outputtz.extend(outputs[0][:, 1])
        result = byz if sum(outputbyz)>sum(outputtz) else tz
        a = ['Path:'+img_path+\
        '\nProbability:'+str(round(math.exp(max(sum(outputbyz),sum(outputtz))/len(blocks))/(math.exp(max(sum(outputbyz),\
        sum(outputtz))/len(blocks))+math.exp(min(sum(outputbyz),sum(outputtz))/len(blocks)))*100,2))+'%'+'\nPrediction Result:'+str(result)]
        # print(result_Image.info['dpi'] if 'dpi' in result_Image.info else None)

        return a, h, w, os.path.basename(img_path).split('.')[0], os.path.basename(img_path).split('.')[1], self.get_file_size(img_path), \
            img.info['dpi'] if 'dpi' in img.info else None, str(result)
            # self.labelpredictone.setText('Path:'+img_path+'\nPredict Result:'+str(byz if sum(outputbyz)>sum(outputtz) else tz)+'\nProbability:'+str(round(math.exp(max(sum(outputbyz),sum(outputtz)))/(math.exp(max(sum(outputbyz),sum(outputtz)))+math.exp(min(sum(outputbyz),sum(outputtz))))*100,2))+'%')

    def predict_all(self, fileName):
        model_path = self.model
        filename = fileName
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        classes = ['byz', 'tz']
        counter = {c: 0 for c in classes}
        other = {o: 0 for o in classes}
        byz = 'Corypha umbraculifera'
        tz = 'Borassus flabellifer'

        # self.tableWidget.setRowCount(len(folder_path))
        
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.JPG') or filename.endswith('.bmp'):
            img_path = filename
            # with open(img_path,"rb" ) as f:
            #     t = f.read()
            # result_a = remove(data=t)
            # result_Image = Image.open(io.BytesIO(result_a)).convert('RGB')
            result_Image = Image.open(img_path).convert('RGB')

            h, w= result_Image.size
            blocks = self.crop_image(result_Image)
            results = []
            outputbyz = []
            outputtz = []
            probbyz = []
            probtz = []
            # total = 3478
            # probbyz = []
            # probtz = []
            batch_size = min(64, len(blocks))  # Batch size does not exceed 64
            for i in range(0, len(blocks), batch_size):
                batch = blocks[i:i + batch_size]
                batch = stack([self.preprocess(block) for block in batch])
                input_name = session.get_inputs()[0].name
                # 运行模型
                outputs = session.run(None, {input_name: batch})
                # predictions = outputs[0]
                outputbyz.extend(outputs[0][:, 0])
                outputtz.extend(outputs[0][:, 1])
            prob_byz = round(math.exp(max(sum(outputbyz),sum(outputtz))/len(blocks))/(math.exp(max(sum(outputbyz),sum(outputtz))/len(blocks))+math.exp(min(sum(outputbyz),sum(outputtz))/len(blocks)))*100,2)
            prob_tz = 100-prob_byz
            df = pd.DataFrame({
            'filename': img_path,
            'prob_byz':[str(prob_byz)+'%'],
            'prob_tz':[str(prob_tz)+'%'],
            'result':[byz if sum(outputbyz)>sum(outputtz) else tz]
            }) 
            return img_path, max(prob_byz,prob_tz), 0 if sum(outputbyz)>sum(outputtz) else 1, h, w
        
    def predict_all_rembg(self, fileName):
        model_path = self.model
        filename = fileName
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])



        classes = ['byz', 'tz']
        counter = {c: 0 for c in classes}
        other = {o: 0 for o in classes}
        byz = 'Corypha umbraculifera'
        tz = 'Borassus flabellifer'

        # self.tableWidget.setRowCount(len(folder_path))
        

        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.JPG') or filename.endswith('.bmp'):
            img_path = filename
            with open(img_path,"rb" ) as f:
                t = f.read()
            result_a = remove(data=t)
            result_Image = Image.open(io.BytesIO(result_a)).convert('RGB')
            h, w= result_Image.size
            blocks = self.crop_image_rembg(result_Image)
            results = []
            outputbyz = []
            outputtz = []
            probbyz = []
            probtz = []
            # total = 3478
            # probbyz = []
            # probtz = []
            batch_size = min(64, len(blocks))  # Batch size does not exceed 64
            for i in range(0, len(blocks), batch_size):
                batch = blocks[i:i + batch_size]
                batch = stack([self.preprocess(block) for block in batch])
                input_name = session.get_inputs()[0].name
                # 运行模型
                outputs = session.run(None, {input_name: batch})
                # predictions = outputs[0]
                outputbyz.extend(outputs[0][:, 0])
                outputtz.extend(outputs[0][:, 1])

            prob_byz = round(math.exp(max(sum(outputbyz),sum(outputtz))/len(blocks))/(math.exp(max(sum(outputbyz),sum(outputtz))/len(blocks))+math.exp(min(sum(outputbyz),sum(outputtz))/len(blocks)))*100,2)
            prob_tz = 100-prob_byz
            df = pd.DataFrame({
            'filename': img_path,
            'prob_byz':[str(prob_byz)+'%'],
            'prob_tz':[str(prob_tz)+'%'],
            'result':[byz if sum(outputbyz)>sum(outputtz) else tz]
            }) 
            return img_path, max(prob_byz,prob_tz), 0 if sum(outputbyz)>sum(outputtz) else 1, h, w

    def init_setmenus(self):
        # 创建分辨率子菜单
        self.resolution_menu = QMenu('Resolution', self)
        # 添加分辨率选项
        for res in ['640*320', 
                    '1280*640', 
                    '2560*1280', 
                    '1000*600']:
            action = QAction(res, self)
            action.setCheckable(True)
            # 默认分辨率为400*400
            if res == '1000*600':
                action.setChecked(True)
            action.triggered.connect(lambda checked, res=res: self.change_resolution(res))
            self.resolution_menu.addAction(action)
            self.resolution_actions.append(action)

        # 创建背景颜色子菜单
        self.color_menu = QMenu('Background color', self)
        # 添加颜色选项
        for color in ['default', 
                      'black', 
                      'blue', 
                      'transparent']:
            action = QAction(color, self)
            action.setCheckable(True)
            # 默认背景颜色为默认
            if color == 'default':
                action.setChecked(True)
            action.triggered.connect(lambda checked, color=color: self.change_color(color))
            self.color_menu.addAction(action)
            self.color_actions.append(action)

        # 创建是否去背景子菜单
        self.rembg_menu = QMenu('Auto remove background', self)
            # 添加选项
        for arb in ['Yes', 
                    'No']:
            action = QAction(arb, self)
            action.setCheckable(True)
            # 去背景为默认
            if arb == 'No':
                action.setChecked(True)
            action.triggered.connect(lambda checked, arb=arb: self.change_arb(arb))
            self.rembg_menu.addAction(action)
            self.arb_actions.append(action)

        # 创建选择模型子菜单
        self.model_menu = QMenu('Model', self)
            # 添加选项
        for model in ['EfficientNet', 
                      'DenseNet', 
                      'ResNet', 
                      'Your own model']:
            action = QAction(model, self)
            action.setCheckable(True)
            # 去背景为默认
            if model == 'EfficientNet':
                action.setChecked(True)
            action.triggered.connect(lambda checked, model=model: self.change_model(model))
            self.model_menu.addAction(action)
            self.model_actions.append(action)

    def open_setmenu(self):
        # 创建菜单
        setmenu = QMenu()

        # 将子菜单添加到主菜单
        setmenu.addMenu(self.resolution_menu)
        setmenu.addMenu(self.color_menu)
        setmenu.addMenu(self.rembg_menu)
        setmenu.addMenu(self.model_menu)

        # 在设置按钮下方显示菜单
        # menu.exec_(self.settings_button.mapToGlobal(self.settings_button.pos()))
        button_geometry = self.settings_button.geometry()
        # print(self.settings_button.geometry())
        setmenu.exec_(self.settings_button.mapToGlobal(QPoint(0, 29)))
        # print(button_geometry.bottomLeft())
   
    def change_resolution(self, resolution):
        # 改变窗口分辨率
        width, height = map(int, resolution.split('*'))
        self.setFixedSize(width, height)
        # 更新菜单中的选中状态
        self.update_checked_status(self.resolution_actions, resolution)

    def change_color(self, color):
        # 改变窗口背景颜色
        if color == 'default':
            self.setStyleSheet("")
        elif color == 'black':
            self.setStyleSheet("background-color: black;")
        elif color == 'blue':
            self.setStyleSheet("background-color: blue;")
        elif color == 'transparent':
            self.setStyleSheet("background-color: transparent;")
        # 更新菜单中的选中状态
        self.update_checked_status(self.color_actions, color)

    def change_arb(self, arb):
        # 改变是否去背景
        if arb == 'Yes':
            self.predict_predict = self.predict_predict_rembg
            self.predict_all = self.predict_all_rembg
        elif arb == 'No':
            self.predict_predict = self.predict_predict
            self.predict_all = self.predict_all
        # 更新菜单中的选中状态
        self.update_checked_status(self.arb_actions, arb)

    def change_model(self, model):
        # 改变软件选择的预测模型
        if model == 'EfficientNet':
            model = self.mymodels['EfficientNet']
            self.predict_predict = lambda fileName:predict_model().predict_predict(fileName,model=model)
            self.predict_all = lambda fileName:predict_model().predict_all(fileName,model=model)
            self.predict_rembg = lambda fileName:predict_model().predict_predict_rembg(fileName,model=model)
            self.predict_all_rembg = lambda fileName:predict_model().predict_all_rembg(fileName,model=model)
        elif model == 'DenseNet':
            model = self.mymodels['DenseNet']
            self.predict_predict = lambda fileName:predict_model().predict_predict(fileName,model=model)
            self.predict_all = lambda fileName:predict_model().predict_all(fileName,model=model)
            self.predict_rembg = lambda fileName:predict_model().predict_predict_rembg(fileName,model=model)
            self.predict_all_rembg = lambda fileName:predict_model().predict_all_rembg(fileName,model=model)
        elif model == 'ResNet':
            model = self.mymodels['ResNet']
            self.predict_predict = lambda fileName:predict_model().predict_predict(fileName,model=model)
            self.predict_all = lambda fileName:predict_model().predict_all(fileName,model=model)
            self.predict_rembg = lambda fileName:predict_model().predict_predict_rembg(fileName,model=model)
            self.predict_all_rembg = lambda fileName:predict_model().predict_all_rembg(fileName,model=model)
        elif model == 'Your own model':
            model = self.mymodels['Your own model']
            self.predict_predict = lambda fileName:predict_model().predict_predict(fileName,model=model)
            self.predict_all = lambda fileName:predict_model().predict_all(fileName,model=model)
            self.predict_rembg = lambda fileName:predict_model().predict_predict_rembg(fileName,model=model)
            self.predict_all_rembg = lambda fileName:predict_model().predict_all_rembg(fileName,model=model)
        # 更新菜单中的选中状态
        self.update_checked_status(self.model_actions, model)

    def update_checked_status(self, actions, selected_action_text):
        # 更新菜单中的选中状态
        for action in actions:
            action.setChecked(action.text() == selected_action_text)

    def open_toolmenu(self):
        # 创建菜单
        toolmenu = QMenu()

        # 创建模型训练子菜单
        trainmodel_menu = QMenu('Model Training', self)
        # 添加模型训练选项
        action_EfficientNet = QAction('EfficientNet', self)
        action_EfficientNet.triggered.connect(self.train_model_EfficientNet)
        trainmodel_menu.addAction(action_EfficientNet)
        action_DenseNet = QAction('DenseNet', self)
        action_DenseNet.triggered.connect(self.train_model_DenseNet)
        trainmodel_menu.addAction(action_DenseNet)
        action_ResNet = QAction('ResNet', self)
        action_ResNet.triggered.connect(self.train_model_ResNet)
        trainmodel_menu.addAction(action_ResNet)
        action_othermodel = QAction('Other model', self)
        action_othermodel.triggered.connect(self.train_model_othermodel)
        trainmodel_menu.addAction(action_othermodel)

        # 创建棕榈叶手稿介绍子菜单
        plant_menu = QMenu('PLM Introduction', self)
        # 添加介绍选项
        action_Borassus = QAction('Borassus', self)
        action_Borassus.triggered.connect(self.openBorassus)
        plant_menu.addAction(action_Borassus)
        action_Corypha = QAction('Corypha', self)
        action_Corypha.triggered.connect(self.openCorypha)
        plant_menu.addAction(action_Corypha)

        # 将子菜单添加到主菜单
        toolmenu.addMenu(trainmodel_menu)
        toolmenu.addMenu(plant_menu)

        # 在设置按钮下方显示菜单
        toolmenu.exec_(self.tools_button.mapToGlobal(QPoint(0, 29)))

    def train_model_EfficientNet(self):
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Function prompt")
        msg_box.setText("This feature is awaiting development")
        msg_box.setIcon(QMessageBox.Information)
        msg_box.exec_()
    def train_model_DenseNet(self):
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Function prompt")
        msg_box.setText("This feature is awaiting development")
        msg_box.setIcon(QMessageBox.Information)
        msg_box.exec_()    
    def train_model_ResNet(self):
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Function prompt")
        msg_box.setText("This feature is awaiting development")
        msg_box.setIcon(QMessageBox.Information)
        msg_box.exec_()
    def train_model_othermodel(self):
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Function prompt")
        msg_box.setText("This feature is awaiting development")
        msg_box.setIcon(QMessageBox.Information)
        msg_box.exec_()
    def openBorassus(self):
        os.startfile(self.introduceBorassus)

    def openCorypha(self):
        os.startfile(self.introduceCorypha)

if __name__ == '__main__':            
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec())