# #-------------------------一般-----------------------------------------
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('QYOLOv11yaml/碩/Attention/0.5RepViTm0_6+QLSKA11+QQiRMB+Concat_BiFPN.yaml')# 如何切换模型版本,
    # model.load('yolov8n.pt') # 是否加载预训练权重,科研不建议大家加载否则很难提升精度
    model.train(data=r"1/data.yaml",
                imgsz=640,
                epochs=100,
                single_cls=False,  # 是否是单类别检测
                batch=16,
                close_mosaic=0,
                optimizer='SGD',
                # resume='runs/train/exp21/weights/last.pt', # 如过想续训就设置last.pt的地址
                amp=True,  # 如果出现训练损失为Nan可以关闭amp
                project='runs/1/QYOLOv11/123',
                name='0.5RepViTm0_6+QLSKA11+QQiRMB+Concat_BiFPN',
                )
#-------------------------一般-----------------------------------------
#-------------------------4185-----------------------------------------
# import warnings
# warnings.filterwarnings('ignore')
# from ultralytics import YOLOc
 
# if __name__ == '__main__':
#     model = YOLO('ultralytics/cfg/models/v8/yolov8.yaml')# 如何切换模型版本,
#     # model.load('yolov8n.pt') # 是否加载预训练权重,科研不建议大家加载否则很难提升精度
#     model.train(data=r"datasets4185/data.yaml",
#                 imgsz=640,
#                 epochs=100,
#                 single_cls=False,  # 是否是单类别检测
#                 batch=16,
#                 close_mosaic=0,
#                 optimizer='SGD',
#                 # resume='runs/train/exp21/weights/last.pt', # 如过想续训就设置last.pt的地址
#                 amp=True,  # 如果出现训练损失为Nan可以关闭amp
#                 project='runs/4185',
#                 name='exp',
#                 )
#-------------------------4185-----------------------------------------
#-------------------------3539-----------------------------------------
# import warnings
# warnings.filterwarnings('ignore')
# from ultralytics import YOLO
 
# if __name__ == '__main__':
#     model = YOLO('ultralytics/cfg/models/11/C3K2/C3k2-AKConv.yaml')# 如何切换模型版本,
#     # model.load('yolov8n.pt') # 是否加载预训练权重,科研不建议大家加载否则很难提升精度
#     model.train(data=r"datasets3539/data.yaml",
#                 imgsz= 640,
#                 epochs=100,
#                 single_cls=False,  # 是否是单类别检测
#                 batch=16,
#                 close_mosaic=0,
#                 optimizer='SGD',
#                 # resume='runs/train/exp21/weights/last.pt', # 如过想续训就设置last.pt的地址
#                 amp=True,  # 如果出现训练损失为Nan可以关闭amp
#                 project='runs/3539',
#                 name='exp',
#                 )
# #-------------------------3539-----------------------------------------