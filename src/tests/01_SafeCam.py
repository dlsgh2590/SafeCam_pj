# 영상 불러와서 프레임 출력 TEST

import cv2
import os

video_path = '../ee.mp4'
cap = cv2.VideoCapture(video_path)

os.makedirs('frames', exist_ok=True)

count = 0
while cap.isOpened() and count < 10:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imwrite(f'frames/frame_{count}.jpg', frame)
    print(f"frame_{count}.jpg 저장됨")
    count += 1

cap.release()
print("완료: 총", count, "개의 프레임 저장됨.")