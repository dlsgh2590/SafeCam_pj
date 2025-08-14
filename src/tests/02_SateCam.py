# 화면 중앙에 사각형(위험구역) 표시

import cv2
import os

# 영상 파일 경로 설정
video_path = '../ee.mp4'
cap = cv2.VideoCapture(video_path)

# 저장 폴더 생성
os.makedirs('frames_with_rect', exist_ok=True)

# 영상에서 첫 프레임 읽기
ret, frame = cap.read()
if not ret:
    print("영상 로딩 실패")
    exit()

# 프레임 크기 확인 (높이, 너비)
height, width = frame.shape[:2]

# 중앙에 위치한 사각형 영역 좌표 계산 (드래그로 위험구역 설정하고 싶지만 지금은 불가능)
box_w, box_h = 200, 200  # 사각형 너비와 높이 (원하는 크기로 조정 가능)
x1 = width // 2 - box_w // 2  # 사각형 좌상단 X
y1 = height // 2 - box_h // 2  # 사각형 좌상단 Y
x2 = x1 + box_w  # 사각형 우하단 X
y2 = y1 + box_h  # 사각형 우하단 Y

# 영상 처음부터 다시 시작
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# 프레임을 하나씩 읽으며 사각형을 그리고 저장 (10프레임만 저장)
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