import cv2

from ultralytics import YOLO

image = cv2.imread('img.png')

model = YOLO('yolov8n.pt')

results = model(image)

for result in results:
    annotated_image = result.plot()

cv2.imshow('Object', annotated_image)

cv2.waitKey(0)

cv2.destroyAllWindows()