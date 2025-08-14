# 3단계: 움직임 감지 (사각형 안만)

import cv2
import os
from ultralytics import YOLO

# YOLO11 모델 불러오기 (COCO pretrained)
model = YOLO('../yolo11n.pt')  # 저번 수업때 썻던 커스텀yolo 불러옴

# 영상 경로
video_path = '../ee.mp4'
cap = cv2.VideoCapture(video_path)

# 저장 폴더
os.makedirs('filtered_frames', exist_ok=True)

# 첫 프레임 크기 확인
ret, frame = cap.read()
if not ret:
    print("영상 로딩 실패")
    exit()

height, width = frame.shape[:2]

# 중앙 위험구역 사각형 좌표 설정 (400x400)
box_w, box_h = 400, 400
x1 = width // 2 - box_w // 2
y1 = height // 2 - box_h // 2
x2 = x1 + box_w
y2 = y1 + box_h

# 영상 처음부터
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

frame_count = 0
save_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO로 객체 탐지
    results = model(frame)[0]

    detected = False

    # 탐지된 객체 반복
    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]

        # 'person' 또는 'motorbike'만 감지
        if label in ['person', 'motorbike']:
            x, y, w, h = box.xyxy[0]
            cx = int((x + w) / 2)
            cy = int((y + h) / 2)

            # 객체 중심점이 위험구역 안에 들어올 경우
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                detected = True
                break

    # 탐지 시 이미지 저장
    if detected:
        print("침입 감지!")
        filename = f'filtered_frames/frame_{save_count}.jpg'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite(filename, frame)
        print(f"→ 저장됨: {filename}")
        save_count += 1

    frame_count += 1

cap.release()
print("완료: 탐지된 객체 저장 완료.")