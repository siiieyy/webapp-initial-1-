from ultralytics import YOLO
import cv2

model = YOLO("E:\\Activities (School)\\Codes\\PD\\detection\\best.pt")

def detect_open_beak_from_frame(frame):
    results = model(frame, verbose=False)
    open_beak_count = 0
    annotated_frame = frame.copy()

    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                print("Detected label:", label)  # 👈 DEBUG THIS
                confidence = float(box.conf[0])

                # 🔥 Fix this comparison
                if "open-mouth" in label.lower():  # check if it contains "open_beak"
                    open_beak_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return open_beak_count, annotated_frame

