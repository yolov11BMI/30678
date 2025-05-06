from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QImage
import cv2
import time
from ultralytics import YOLO

class MWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO 實時辨識 (含FPS)")
        self.resize(1280, 720)

        self.model = YOLO(r"yolo11\weights\best_ncnn_model")  # 請確認權重路徑正確

        self.initUI()
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.prev_time = time.time()
        self.fps = 0

    def initUI(self):
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet('border: 3px solid #D7E2F9;')

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)  # 水平翻轉比較像自拍

        # YOLO推論
        results = self.model.predict(source=frame, verbose=False)
        annotated_frame = results[0].plot()

        # FPS 計算
        current_time = time.time()
        elapsed_time = current_time - self.prev_time
        self.prev_time = current_time
        self.fps = 1.0 / elapsed_time

        # 顯示 FPS
        cv2.putText(annotated_frame, f"FPS: {self.fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 顯示到 QLabel
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def closeEvent(self, event):
        if self.cap.isOpened():
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = MWindow()
    window.show()
    sys.exit(app.exec())
