from PySide6 import QtWidgets, QtCore, QtGui
import cv2, os, time
from threading import Thread
import openai
from ultralytics import YOLO

# 避免每次 YOLO 處理都輸出調試信息
os.environ['YOLO_VERBOSE'] = 'False'

class MWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUI()
        
        # 按鈕綁定
        self.camBtn.clicked.connect(self.startCamera)
        self.stopBtn.clicked.connect(self.stop)
        self.importBtn.clicked.connect(self.loadModel)
        self.folderBtn.clicked.connect(self.loadFolder)
        self.videoBtn.clicked.connect(self.playImageVideo)
        self.nextBtn.clicked.connect(self.nextImage)
        self.prevBtn.clicked.connect(self.prevImage)
        self.generateReportBtn.clicked.connect(self.generateBMIReport)
        
        # 定義定時器，控制顯示視頻的幀率
        self.timer_camera = QtCore.QTimer()
        self.timer_camera.timeout.connect(self.show_camera)
        
        # 新增：倒數計時器
        self.countdown_timer = QtCore.QTimer()
        self.countdown_timer.timeout.connect(self.updateCountdown)
        self.countdown = 3
        self.is_counting = False
        self.snapshot = None  # 儲存截圖

        # 初始化 YOLO 模型和相關參數
        self.model = None
        self.frameToAnalyze = []
        self.images = []
        self.currentImageIndex = -1
        self.cap = None
        Thread(target=self.frameAnalyzeThreadFunc, daemon=True).start()

    def setupUI(self):
        self.resize(1200, 800)
        self.setWindowTitle('YOLO BMI 檢測和健康餐點建議系統')
        
        # 中央部件和主佈局
        centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(centralWidget)
        mainLayout = QtWidgets.QVBoxLayout(centralWidget)

        # 圖形展示部分
        topLayout = QtWidgets.QHBoxLayout()
        self.label_ori_video = QtWidgets.QLabel(self)
        self.label_treated = QtWidgets.QLabel(self)
        self.countdownLabel = QtWidgets.QLabel("")
        self.countdownLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.countdownLabel.setStyleSheet("QLabel { color: red; font-size: 48px; }")
        
        self.label_ori_video.setMinimumSize(320, 320)
        self.label_treated.setMinimumSize(320, 320)
        self.label_ori_video.setStyleSheet('border:1px solid #D7E2F9;')
        self.label_treated.setStyleSheet('border:1px solid #D7E2F9;')
        
        # 創建左側垂直佈局來包含原始視頻和倒數計時
        leftLayout = QtWidgets.QVBoxLayout()
        leftLayout.addWidget(self.label_ori_video)
        leftLayout.addWidget(self.countdownLabel)
        
        topLayout.addLayout(leftLayout)
        topLayout.addWidget(self.label_treated)
        mainLayout.addLayout(topLayout)

        # 下半部分：按鈕和文本顯示
        groupBox = QtWidgets.QGroupBox(self)
        bottomLayout = QtWidgets.QHBoxLayout(groupBox)
        self.textLog = QtWidgets.QTextBrowser()
        bottomLayout.addWidget(self.textLog)
        mainLayout.addWidget(groupBox)

        # 按鈕佈局
        btnLayout = QtWidgets.QVBoxLayout()
        self.importBtn = QtWidgets.QPushButton('📂載入模型')
        self.folderBtn = QtWidgets.QPushButton('📁載入圖片文件夾')
        self.videoBtn = QtWidgets.QPushButton('🎞️圖片影片')
        self.camBtn   = QtWidgets.QPushButton('📽鏡頭')
        self.stopBtn  = QtWidgets.QPushButton('❌停止')
        self.prevBtn  = QtWidgets.QPushButton('⬅️上一張')
        self.nextBtn  = QtWidgets.QPushButton('➡️下一張')
        self.generateReportBtn = QtWidgets.QPushButton('📄生成健康餐點建議')
        self.apiKeyInput = QtWidgets.QLineEdit()
        self.apiKeyInput.setPlaceholderText("輸入 OpenAI API Key")
        
        # 添加按鈕到佈局
        btnLayout.addWidget(self.importBtn)
        btnLayout.addWidget(self.folderBtn)
        btnLayout.addWidget(self.videoBtn)
        btnLayout.addWidget(self.camBtn)
        btnLayout.addWidget(self.stopBtn)
        btnLayout.addWidget(self.prevBtn)
        btnLayout.addWidget(self.nextBtn)
        btnLayout.addWidget(self.generateReportBtn)
        btnLayout.addWidget(self.apiKeyInput)
        bottomLayout.addLayout(btnLayout)

    def loadModel(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, '選擇模型權重文件', '', 'Model Files (*.pt)')
        if file_name:
            self.model = YOLO(file_name)
            self.textLog.append(f"已加載模型權重: {file_name}")

    def loadFolder(self):
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "選擇圖片文件夾")
        if folder_path:
            # 仅加载指定格式的图像文件
            self.images = [os.path.join(folder_path, img) for img in os.listdir(folder_path) 
                           if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if self.images:
                self.currentImageIndex = 0
                self.showImage()
                self.textLog.append(f"已加載圖片文件夾: {folder_path}")
                self.textLog.append(f"圖片列表: {self.images}")
            else:
                self.textLog.append("選定的文件夾中沒有圖像文件。")

    def showImage(self):
        if 0 <= self.currentImageIndex < len(self.images):
            img_path = self.images[self.currentImageIndex]
            frame = cv2.imread(img_path)
            if frame is not None:
                frame = cv2.resize(frame, (320, 320))
                self.frameToAnalyze.append(frame)
                self.displayImage(self.label_ori_video, frame)
            else:
                self.textLog.append(f"無法加載圖片：{img_path}")

    def nextImage(self):
        if self.currentImageIndex < len(self.images) - 1:
            self.currentImageIndex += 1
            self.showImage()

    def prevImage(self):
        if self.currentImageIndex > 0:
            self.currentImageIndex -= 1
            self.showImage()

    def playImageVideo(self):
        if not self.images:
            self.textLog.append("請先載入圖片文件夾。")
            return
        
        self.currentImageIndex = 0
        self.image_video_timer = QtCore.QTimer()
        self.image_video_timer.timeout.connect(self.nextImageForVideo)
        self.image_video_timer.start(500)  # 每 500 毫秒切換一張圖片

    def nextImageForVideo(self):
        if self.currentImageIndex < len(self.images) - 1:
            self.currentImageIndex += 1
            self.showImage()
        else:
            self.image_video_timer.stop()
            self.textLog.append("圖片影片播放結束。")

    def startCamera(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self.cap.isOpened():
            self.textLog.append("鏡頭無法開啟")
            return
        
        # 開始倒數
        self.countdown = 3
        self.is_counting = True
        self.countdownLabel.setText(str(self.countdown))
        self.countdown_timer.start(1000)  # 每秒更新一次
        
        if not self.timer_camera.isActive():
            self.timer_camera.start(50)

    def updateCountdown(self):
        self.countdown -= 1
        if self.countdown >= 0:
            self.countdownLabel.setText(str(self.countdown))
        else:
            self.countdown_timer.stop()
            self.countdownLabel.setText("")
            self.is_counting = False
            # 倒數結束時拍照
            self.takePhoto()

    def takePhoto(self):
        if self.cap is None or not self.cap.isOpened():
            return
            
        success, frame = self.cap.read()
        if success:
            # 裁剪中心 320x320 區域
            height, width = frame.shape[:2]
            center_x, center_y = width // 2, height // 2
            crop_size = 160
            cropped_frame = frame[center_y - crop_size:center_y + crop_size, 
                                center_x - crop_size:center_x + crop_size]
            
            # 將裁剪的圖像縮放到 640x640
            resized_frame = cv2.resize(cropped_frame, (640, 640))
            
            # 儲存截圖並進行分析
            self.snapshot = resized_frame.copy()
            results = self.model.predict(source=self.snapshot, iou=0.25)  # 調整 IoU 門檻
            if results and results[0].boxes:
                boxes = results[0].boxes
                best_box = max(boxes, key=lambda box: box.conf.item())  # 保留置信度最高的框
                img = results[0].plot(line_width=1)  # 繪製整個結果
                self.displayImage(self.label_treated, img)  # 更新右側處理後的結果顯示
                
                # 根據 YOLO 檢測結果生成 BMI 類別
                labels = [box.label for box in boxes]
                if 'overweight' in labels:
                    bmi_category = "過重"
                elif 'underweight' in labels:
                    bmi_category = "過輕"
                else:
                    bmi_category = "正常體重"
                
                # 生成健康餐點建議
                self.generateBMIReport(bmi_category)
            else:
                self.textLog.append("沒有檢測到任何對象")
            
            # 關閉鏡頭
            self.stop()
            self.textLog.append("已完成拍照和分析")

    def show_camera(self):
        if not self.cap or not self.cap.isOpened():
            return
            
        success, frame = self.cap.read()
        if not success:
            return

        # 裁剪中心 320x320 區域
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        crop_size = 160
        cropped_frame = frame[center_y - crop_size:center_y + crop_size, 
                            center_x - crop_size:center_x + crop_size]

        # 將裁剪的圖像縮放到 640x640
        resized_frame = cv2.resize(cropped_frame, (640, 640))
        
        # 更新攝像頭畫面
        self.displayImage(self.label_ori_video, resized_frame)

    def frameAnalyzeThreadFunc(self):
        while True:
            if not self.frameToAnalyze:
                time.sleep(0.01)
                continue
            frame = self.frameToAnalyze.pop(0)
            if self.model:
                results = self.model(frame, iou=0.3)[0]
                if results.boxes:
                    img = results.plot(line_width=1)  # 繪製整個結果，而不是直接使用 Boxes
                    QtCore.QTimer.singleShot(0, lambda: self.displayImage(self.label_treated, img))
                time.sleep(0.5)

    def displayImage(self, label, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qImage = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
        label.setPixmap(QtGui.QPixmap.fromImage(qImage))

    def generateBMIReport(self, bmi_category=None):
        api_key = self.apiKeyInput.text().strip()
        if not api_key:
            self.textLog.append("請輸入有效的 API Key")
            return
        openai.api_key = api_key

        # 根據 BMI 類別生成健康餐點建議
        if not bmi_category:
            bmi_category = "正常體重"
        prompt = f"根據 {bmi_category} 提供適合的健康餐點建議，請包含三餐的營養建議。"
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "system", "content": prompt}]
            )
            meal_suggestion = response.choices[0].message['content']
            self.textLog.append(f"健康餐點建議: {meal_suggestion}")
        except Exception as e:
            self.textLog.append(f"生成健康餐點建議時出錯: {e}")

    def stop(self):
        if hasattr(self, 'timer_camera') and self.timer_camera.isActive():
            self.timer_camera.stop()
        if hasattr(self, 'countdown_timer') and self.countdown_timer.isActive():
            self.countdown_timer.stop()
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        self.label_ori_video.clear()
        if self.snapshot is None or not self.snapshot.any():  # 只有在沒有截圖時才清除右側畫面
            self.label_treated.clear()
        self.countdownLabel.clear()
        self.is_counting = False

# 確保 QApplication 和主窗口代碼在程序的入口處運行
if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MWindow()
    window.show()
    app.exec()
