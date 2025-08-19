import cv2
import os
import time
from ultralytics import YOLO
from datetime import timedelta

# YOLO 모델 로드
model = YOLO('../yolov8n.pt')

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

# 위험구역 그리기 위한 변수들 준비
drawing = False  # 마우스 드래그 상태 확인
ix, iy = -1, -1  # 드래그 시작점 좌표
x1, y1, x2, y2 = 0, 0, 0, 0  # 위험구역 좌상단, 우하단 좌표

# 마우스로 사각형 그리기 함수
def draw_rectangle(event, x, y, flags, param):
    global drawing, ix, iy, x1, y1, x2, y2, frame
    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 왼쪽 버튼 누름
        drawing = True
        ix, iy = x, y  # 시작 위치 저장
    elif event == cv2.EVENT_MOUSEMOVE:  # 마우스 움직임
        if drawing:
            temp_frame = frame.copy()  # 원본 복사
            cv2.rectangle(temp_frame, (ix, iy), (x, y), (0, 255, 0), 2)  # 초록색 사각형 그리기
            cv2.imshow("Set Danger Zone", temp_frame)  # 화면에 보여주기
    elif event == cv2.EVENT_LBUTTONUP:  # 마우스 왼쪽 버튼 뗌
        drawing = False
        # 사각형 좌표 계산 (왼쪽 위, 오른쪽 아래)
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        # 빨간색 사각형으로 위험구역 표시
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow("Set Danger Zone", frame)

# 위험구역 설정 화면 만들기
cv2.namedWindow("Set Danger Zone")
cv2.setMouseCallback("Set Danger Zone", draw_rectangle)
print("마우스로 위험구역을 드래그해서 지정하고 's' 키를 누르세요.")

# 위험구역 지정할 때까지 계속 화면 보여주기
while True:
    cv2.imshow("Set Danger Zone", frame)
    if cv2.waitKey(1) == ord('s'):
        break

cv2.destroyWindow("Set Danger Zone")

# 영상 처음부터 다시 시작
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

yolo_interval = 3  # 몇 프레임마다 YOLO 실행할지 설정
frame_count = 0  # 프레임 번호
save_count = 0  # 저장된 사진 번호

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 초록 위험구역 표시

    person_motorbike_in_zone = False
    construction_vehicle_in_zone = False 

    if frame_count % yolo_interval == 0:
        results = model(frame, verbose=False)[0]  # 물체 찾기

        for box in results.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            x1_box, y1_box, x2_box, y2_box = box.xyxy[0]
            cx = int((x1_box + x2_box) / 2)
            cy = int((y1_box + y2_box) / 2)

            if x1 <= cx <= x2 and y1 <= cy <= y2:
                if label in ['person', 'motorbike']:
                    person_motorbike_in_zone = True
                elif label in ['truck', 'trailer', 'bus', 'construction vehicle', 'car']:
                    construction_vehicle_in_zone = True

        if person_motorbike_in_zone and not construction_vehicle_in_zone:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 빨간 네모

            current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            time_str = str(timedelta(milliseconds=current_time_ms)).split('.')[0]
            time_text = f'Time: {time_str}'

            cv2.putText(frame, "Intrusion Detected", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, time_text, (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            filename = f'filtered_frames/frame_{save_count}_{time_str.replace(":", "-")}.jpg'
            cv2.imwrite(filename, frame)  # 사진 저장
            print(f"[Alert] Intrusion detected - photo saved: {filename}")
            save_count += 1

    if person_motorbike_in_zone and not construction_vehicle_in_zone:
        cv2.putText(frame, "Warning! Intruder detected", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)  # 경고 문구

    cv2.imshow("Live Detection", frame)  # 화면 보여주기

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 누르면 종료
        break

    frame_count += 1  # 다음 프레임


cap.release()
cv2.destroyAllWindows()
print("모니터링이 종료되었습니다.")
