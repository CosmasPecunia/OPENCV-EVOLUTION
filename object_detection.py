import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

confidence_threshold = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    # On copie la frame pour dessiner dessus
    annotated_frame = frame.copy()
    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf > confidence_threshold:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            cls = int(box.cls[0])
            label = results[0].names[cls]
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated_frame, f"{label} {conf * 100:.2f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.imshow("YOLOv8 - Webcam", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
