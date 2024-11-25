# import pandas as pd  # 匯入 pandas 庫
# import numpy as np  # 匯入 numpy 庫
# import matplotlib.pyplot as plt  # 匯入 matplotlib 庫以進行繪圖

# if __name__ == '__main__':
#     # File paths to CSV results
#     yolov11_result_csv = '/home/eparc3080/yolov11/ultralytics-main test code/runs/detect/train new73/results.csv'  # YOLOv11 結果的 CSV 檔案路徑
#     yoloyolo_result_csv = '/home/eparc3080/yolov11/ultralytics-main test code/runs/detect/C2PSA/new73 C2LSKA 23/results.csv'  # YOLOv11-C2LSKA 結果的 CSV 檔案路徑
#     yolovqyolo_result_csv = '/home/eparc3080/yolov11/ultralytics-main test code/runs/detect/C2PSA/train QC2LSKA(ela)/results.csv'  # YOLOv11-QC2LSKA 結果的 CSV 檔案路徑
    
#     # Load CSV data
#     yolov11_result_data = pd.read_csv(yolov11_result_csv)  # 加載 YOLOv11 結果的 CSV 檔案
#     yoloyolo_result_data = pd.read_csv(yoloyolo_result_csv)  # 加載 YOLOv11-C2LSKA 結果的 CSV 檔案
#     yolovqyolo_result_data = pd.read_csv(yolovqyolo_result_csv)  # 加載 YOLOv11-QC2LSKA 結果的 CSV 檔案
    
#     # Define epoch variable (assuming it corresponds to the length of data in the CSV)
#     epoch8 = len(yolov11_result_data)  # 定義 epoch8，對應於 YOLOv11 結果資料的長度
#     epoch9 = len(yoloyolo_result_data)  # 定義 epoch9，對應於 YOLOv11-C2LSKA 結果資料的長度
#     epoch10 = len(yolovqyolo_result_data)  # 定義 epoch10，對應於 YOLOv11-QC2LSKA 結果資料的長度
    
#     # Set font to Times New Roman for all text elements
#     plt.rcParams['font.family'] = 'Times New Roman'  # 將所有文字元素的字體設定為 Times New Roman
    
#     # Plotting the metrics for YOLO models
#     plt.figure(figsize=(10, 8))  # 設定繪圖的圖形大小為 10x8 英吋
    
#     # Plot the data with custom colors, thicker lines, and labels
#     line1, = plt.plot(np.arange(epoch8), yolov11_result_data['       metrics/mAP50(B)'], label='YOLO11-C2PSA', linewidth=3, color='green')  # 使用自訂顏色和較粗的線繪製 YOLOv11-C2PSA 的數據
#     line2, = plt.plot(np.arange(epoch9), yoloyolo_result_data['       metrics/mAP50(B)'], label='YOLO11-C2LSKA', linewidth=3, color='blue')  # 使用自訂顏色和較粗的線繪製 YOLOv11-C2LSKA 的數據
#     line3, = plt.plot(np.arange(epoch10), yolovqyolo_result_data['       metrics/mAP50(B)'], label='YOLO11-QC2LSKA', linewidth=3, color='red')  # 使用自訂顏色和較粗的線繪製 YOLOv11-QC2LSKA 的數據
    
#     # Set the x-axis and y-axis labels with larger fonts
#     plt.xlabel('Epoch', fontsize=16)  # 設定 x 軸標籤為 'Epoch'，字體大小為 16
#     plt.ylabel('mAP50', fontsize=16)  # 設定 y 軸標籤為 'mAP50'，字體大小為 16
    
#     # Set the legend with custom order (YOLO11-C2QLSKA first) in the lower left corner
#     plt.legend(handles=[line3, line2, line1], fontsize=14, loc='lower left')  # 設定圖例的順序（YOLO11-QC2LSKA 優先），放置於左下角，字體大小為 14
    
#     # Set x-axis and y-axis limits to avoid gaps at the edges
#     # plt.xlim([0, 100])  # 確保 x 軸從 0 開始並在 100 結束
#     # plt.ylim([0, 0.8])  # 根據數據範圍調整 y 軸
    
#     # Tight layout and save the plot
#     plt.tight_layout()  # 自動調整子圖參數以填充整個圖形區域
#     plt.savefig('mAP50-curve.png')  # 保存繪製的圖形為 'mAP50-curve.png'
import pandas as pd  # 匯入 pandas 庫
import numpy as np  # 匯入 numpy 庫
import matplotlib.pyplot as plt  # 匯入 matplotlib 庫以進行繪圖

if __name__ == '__main__':
    # File paths to CSV results
    result1_csv = '碩論repvit/exp/yolov11/results.csv'  
    result2_csv = '碩論LSKA/C2PSA_QLSKA/results.csv'  
    result3_csv = 'runs/1/QYOLOv11/123/0.5RepViT_m0_6+QC2LSKA11/results.csv' 
    result4_csv = 'runs/1/QYOLOv11/123/0.5RepViTm0_6+QLSKA11+QQiRMB(P5)(0.82)/results.png' #
    # result5_csv = 'runs/train/bmiv10 v8/results.csv'  
    # result6_csv = 'runs/train/yolov8s-Qc2f-Qbiformer0.732/results.csv' #
    # result7_csv = 'runs/train/exp/results.csv'  
    # result8_csv = 'runs/train/yolov8s-Qyolo(0.754)/results.csv'
    # Load CSV data
    result1_data = pd.read_csv(result1_csv)  # 加載 YOLOv11 結果的 CSV 檔案
    result2_data = pd.read_csv(result2_csv)  # 加載 YOLOv11-C2LSKA 結果的 CSV 檔案
    result3_data = pd.read_csv(result3_csv)  # 加載 YOLOv11-QC2LSKA 結果的 CSV 檔案
    result4_data = pd.read_csv(result4_csv)
    # result5_data = pd.read_csv(result5_csv)
    # result6_data = pd.read_csv(result6_csv)
    # result7_data = pd.read_csv(result7_csv)
    # result8_data = pd.read_csv(result8_csv)
    # Define epoch variable (assuming it corresponds to the length of data in the CSV)
    epoch1 = len(result1_data)  # 定義 epoch8，對應於 YOLOv11 結果資料的長度
    epoch2 = len(result2_data)  # 定義 epoch9，對應於 YOLOv11-C2LSKA 結果資料的長度
    epoch3 = len(result3_data)  # 定義 epoch10，對應於 YOLOv11-QC2LSKA 結果資料的長度
    epoch4 = len(result4_data)
    # epoch5 = len(result5_data)
    # epoch6 = len(result6_data)
    # epoch7 = len(result7_data)  
    # epoch8 = len(result8_data) 
 

    # Set font to Times New Roman for all text elements
    # plt.rcParams['font.family'] = 'Times New Roman'  # 將所有文字元素的字體設定為 Times New Roman
    
    # Plotting the metrics for YOLO models
    plt.figure(figsize=(10, 8))  # 設定繪圖的圖形大小為 10x8 英吋
    
    # Plot the data with custom colors, thicker lines, and labels
    line1, = plt.plot(np.arange(epoch1), result1_data['       metrics/mAP50(B)'], label='YOLOv11', linewidth=3, color='black')  # 使用自訂顏色和較粗的線繪製 YOLOv11-C2PSA 的數據
    line2, = plt.plot(np.arange(epoch2), result2_data['       metrics/mAP50(B)'], label='YOLOv11-QC2LSKA', linewidth=3, color='purple')  # 使用自訂顏色和較粗的線繪製 YOLOv11-C2LSKA 的數據
    line3, = plt.plot(np.arange(epoch3), result3_data['       metrics/mAP50(B)'], label='YOLOv11-QRepViT_QC2LSKA', linewidth=3, color='indigo')  # 使用自訂顏色和較粗的線繪製 YOLOv11-QC2LSKA 的數據
    line4, = plt.plot(np.arange(epoch4), result4_data['       metrics/mAP50(B)'], label='QYOLOv11', linewidth=3, color='blue')
    # line5, = plt.plot(np.arange(epoch5), result5_data['       metrics/mAP50(B)'], label='QBiFormer w/o', linewidth=3, color='blue')
    # line6, = plt.plot(np.arange(epoch6), result6_data['       metrics/mAP50(B)'], label='QConvNeXt w/o', linewidth=3, color='green')
    # line7, = plt.plot(np.arange(epoch7), result7_data['       metrics/mAP50(B)'], label='QC2f w/o', linewidth=3, color='orange')
    # line8, = plt.plot(np.arange(epoch7), result8_data['       metrics/mAP50(B)'], label='QYOLOv8', linewidth=3, color='red')
    # Set the x-axis and y-axis labels with larger fonts
    plt.xlabel('Epoch', fontsize=16)  # 設定 x 軸標籤為 'Epoch'，字體大小為 16
    plt.ylabel('mAP50', fontsize=16)  # 設定 y 軸標籤為 'mAP50'，字體大小為 16
    
    # Set the legend with custom order (YOLO11-C2QLSKA first) in the lower left corner
    plt.legend(handles=[line8,line5,line6,line7, line1], fontsize=14, loc='lower right')  # 設定圖例的順序（YOLO11-QC2LSKA 優先），放置於左下角，字體大小為 14
    
    # Set x-axis and y-axis limits to avoid gaps at the edges
    plt.xlim([0, 100])  # 確保 x 軸從 0 開始並在 100 結束
    plt.ylim([0, 0.8])  # 根據數據範圍調整 y 軸
    
    # Tight layout and save the plot
    plt.tight_layout()  # 自動調整子圖參數以填充整個圖形區域
    plt.savefig('mAP50-curve.png')  # 保存繪製的圖形為 'mAP50-curve.png'

