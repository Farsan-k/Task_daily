import cv2

from ultralytics import YOLO

cap = cv2.VideoCapture('img.jpg')

model = YOLO('yolov8n.pt')


while True:
    ret, frame = cap.read()

    if not ret:
        print('No frame captured')
        break

    results = model(frame)

    for result in results:
        annotated_image = result.plot()
    cv2.imshow('Objest', annotated_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()