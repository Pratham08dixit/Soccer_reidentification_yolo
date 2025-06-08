import cv2
from ultralytics import YOLO

# Load YOLOv11 model
model = YOLO('best.pt')  # path to your fine-tuned model

# Open video
cap = cv2.VideoCapture('15sec_input_720p.mp4')

# Track ID cache
id_frame_counts = {}
frame_num = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection + tracking (ByteTrack used by default)
    results = model.track(frame, persist=True, conf=0.3, tracker='bytetrack.yaml')

    if results and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        for box, track_id, cls in zip(boxes, ids, classes):
            label = model.names[int(cls)]
            if label != 'player':
                continue

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID:{int(track_id)}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            id_frame_counts[int(track_id)] = id_frame_counts.get(int(track_id), 0) + 1

    # Show the live annotated frame
    cv2.imshow('Live Player Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_num += 1

cap.release()
cv2.destroyAllWindows()
