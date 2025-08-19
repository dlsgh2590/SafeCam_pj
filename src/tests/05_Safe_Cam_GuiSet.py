# GUI 마우스 드래그 방식으로 위험구역 설정

import cv2
import os
from ultralytics import YOLO
from datetime import timedelta

# YOLO 모델 로드
#model = YOLO('../yolo11n.pt')  # 커스텀 YOLO 모델 경로
model = YOLO('../yolov8n.pt') # yolo8 공식모델로 변경

#테스트 결과
#yolov8m.pt	중간 크기, 정확도와 속도 균형 좋음
#yolov8l.pt	높은 정확도 (속도는 약간 느림)
#yolov8x.pt	최고 정확도, GPU 필수

# 영상 경로
video_path = '../ee.mp4'
cap = cv2.VideoCapture(video_path)

# 저장 폴더 생성
os.makedirs('filtered_frames', exist_ok=True)

# 첫 프레임 읽기
ret, frame = cap.read()
if not ret:
    print("영상 로딩 실패")
    exit()

# 위험구역 설정용 변수
drawing = False
ix, iy = -1, -1
x1, y1, x2, y2 = 0, 0, 0, 0

# 마우스 콜백 함수
def draw_rectangle(event, x, y, flags, param):
    global drawing, ix, iy, x1, y1, x2, y2, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_frame = frame.copy()
            cv2.rectangle(temp_frame, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Set Danger Zone", temp_frame)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow("Set Danger Zone", frame)

# 위험구역 설정 GUI
cv2.namedWindow("Set Danger Zone")
cv2.setMouseCallback("Set Danger Zone", draw_rectangle)
print("위험구역을 마우스로 드래그하여 설정한 뒤 's' 키를 눌러 시작하세요.")

while True:
    cv2.imshow("Set Danger Zone", frame)
    key = cv2.waitKey(1)
    if key == ord('s'):
        break

cv2.destroyWindow("Set Danger Zone")

# 영상 처음부터 다시 시작
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

yolo_interval = 3  # YOLO 추론 주기
frame_count = 0
save_count = 0
detected = False
time_text = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % yolo_interval == 0:
        results = model(frame, verbose=False)[0]
        detected = False  # 매 프레임마다 초기화

        for box in results.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label in ['person', 'motorbike', 'truck', 'bus']:
                x, y, w, h = box.xyxy[0]
                cx = int((x + w) / 2)
                cy = int((y + h) / 2)

                if x1 <= cx <= x2 and y1 <= cy <= y2:
                   detected = True
                   break

        # 침입 감지되면 즉시 처리
        if detected:
            # 시간 계산
            current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            time_str = str(timedelta(milliseconds=current_time_ms)).split('.')[0]
            time_text = f'{time_str}'

            # 위험구역 표시
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # 시간 텍스트 표시
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.8
            thickness = 2
            text_size, _ = cv2.getTextSize(time_text, font, scale, thickness)
            text_x = 30
            text_y = 40
            cv2.putText(frame, time_text, (text_x, text_y), font, scale, (0, 0, 255), thickness, cv2.LINE_AA)

            # 이미지 저장
            filename = f'filtered_frames/frame_{save_count}_{time_str.replace(":", "-")}.jpg'
            cv2.imwrite(filename, frame)
            print(f"[!] 침입 감지! 저장됨: {filename}")
            save_count += 1

        # 프레임 표시
        cv2.imshow("Live Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break


    frame_count += 1

cap.release()
cv2.destroyAllWindows()
print("완료: 감지 및 실시간 모니터링 종료.")
