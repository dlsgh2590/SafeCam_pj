# 화면 중앙에 사각형(위험구역) 표시

import cv2
import os

video_path = '../ee.mp4'
cap = cv2.VideoCapture(video_path)

# 저장 폴더 생성
os.makedirs('frames_with_rect', exist_ok=True)

ret, frame = cap.read()
if not ret:
    print("영상 로딩 실패")
    exit()

height, width = frame.shape[:2]
box_w, box_h = 200, 200
x1 = width // 2 - box_w // 2
y1 = height // 2 - box_h // 2
x2 = x1 + box_w
y2 = y1 + box_h

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 처음 프레임부터 다시 시작
count = 0

while cap.isOpened() and count < 10:
    ret, frame = cap.read()
    if not ret:
        break

    # 위험구역 사각형 그리기
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 이미지 저장
    filename = f'frames_with_rect/frame_{count}.jpg'
    cv2.imwrite(filename, frame)
    print(f"저장됨: {filename}")
    count += 1

cap.release()
print("완료: 사각형 표시된 프레임 저장됨.")