import cv2

from ultralytics import YOLO

image = cv2.imread('img.png')

model = YOLO('yolov8n.pt')

