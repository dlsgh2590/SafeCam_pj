# 3단계: 움직임 감지 (사각형 안만)
# + 영상에서 언제 저장된 사진인지 확인하기 위해 시간 추가

import cv2
import os
from ultralytics import YOLO
from datetime import timedelta  # 시간 포맷을 위해 사용

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
    results = model(frame, verbose=False)[0]

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
        # print("침입 감지!")

        # 현재 시간(ms) → HH:MM:SS 포맷으로 변환
        current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        time_str = str(timedelta(milliseconds=current_time_ms)).split('.')[0]  # HH:MM:SS
        time_text = f'Time: {time_str}'

        # 위험구역 사각형 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 오른쪽 하단 텍스트 위치 계산
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(time_text, font, scale, thickness)

        margin = 10
        text_x = frame.shape[1] - text_width - margin
        text_y = frame.shape[0] - margin

        # 시간 텍스트 추가
        cv2.putText(frame, time_text, (text_x, text_y), font, scale, (0, 255, 255), thickness, cv2.LINE_AA)

        # 이미지 저장 (파일명에 시간 포함)
        filename = f'filtered_frames/frame_{save_count}_{time_str.replace(":", "-")}.jpg'
        cv2.imwrite(filename, frame)
        # print(f"→ 저장됨: {filename}")
        save_count += 1

    frame_count += 1

cap.release()
print("완료: 탐지된 객체 저장 완료.")