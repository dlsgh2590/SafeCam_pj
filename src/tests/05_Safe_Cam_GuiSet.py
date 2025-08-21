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
##video_path = '../ee.mp4'
video_path = '../dd.mp4'
cap = cv2.VideoCapture(video_path)

# 저장 폴더 생성
os.makedirs('filtered_frames', exist_ok=True)

# 첫 프레임 읽기
ret, current_frame = cap.read()
if not ret:
    print("영상 로딩 실패")
    exit()

# 위험구역 그리기 위한 변수들 준비
is_drawing = False  # 마우스 드래그 상태 확인
start_x, start_y = -1, -1  # 드래그 시작점 좌표
zone_x1, zone_y1, zone_x2, zone_y2 = 0, 0, 0, 0  # 위험구역 좌상단, 우하단 좌표

# 마우스로 사각형 그리기 함수
def draw_rectangle(event, x, y, flags, param):
    global is_drawing, start_x, start_y, zone_x1, zone_y1, zone_x2, zone_y2, current_frame
    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 왼쪽 버튼 누름
        is_drawing = True
        start_x, start_y = x, y  # 시작 위치 저장
    elif event == cv2.EVENT_MOUSEMOVE:  # 마우스 움직임
        if is_drawing:
            temp_frame = current_frame.copy()  # 원본 복사
            cv2.rectangle(temp_frame, (start_x, start_y), (x, y), (0, 255, 0), 2)  # 초록색 사각형 그리기
            cv2.imshow("Set Danger Zone", temp_frame)  # 화면에 보여주기
    elif event == cv2.EVENT_LBUTTONUP:  # 마우스 왼쪽 버튼 뗌
        is_drawing = False
        # 사각형 좌표 계산 (왼쪽 위, 오른쪽 아래)
        zone_x1, zone_y1 = min(start_x, x), min(start_y, y)
        zone_x2, zone_y2 = max(start_x, x), max(start_y, y)
        # 빨간색 사각형으로 위험구역 표시
        cv2.rectangle(current_frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (0, 0, 255), 2)
        cv2.imshow("Set Danger Zone", current_frame)

# 위험구역 설정 화면 만들기
cv2.namedWindow("Set Danger Zone")
cv2.setMouseCallback("Set Danger Zone", draw_rectangle)
print("마우스로 위험구역을 드래그해서 지정하고 's' 키를 누르세요.")

# 위험구역 지정할 때까지 계속 화면 보여주기
while True:
    cv2.imshow("Set Danger Zone", current_frame)
    if cv2.waitKey(1) == ord('s'):
        break

cv2.destroyWindow("Set Danger Zone")

# 영상 처음부터 다시 시작
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

yolo_interval = 3  # 몇 프레임마다 YOLO 실행할지 설정
frame_count = 0  # 프레임 번호
save_count = 0  # 저장된 사진 번호

# 실시간 경고 상태를 유지하기 위한 변수 추가
show_alert_now = False
#last_alert_time = 0
#alert_duration = 30  # YOLO 간격과 별도로, 몇 프레임 동안 경고 문구 유지할지

while cap.isOpened():
    ret, current_frame = cap.read()
    if not ret:
        break

    cv2.rectangle(current_frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (0, 255, 0), 2)  # 초록 위험구역 표시

    person_motorbike_in_zone = False
    construction_vehicle_in_zone = False 

    # YOLO 실행 주기 맞을 때만 탐지
    if frame_count % yolo_interval == 0:
        results = model(current_frame, verbose=False)[0]  # 물체 찾기

        for box in results.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            x1_box, y1_box, x2_box, y2_box = map(int, box.xyxy[0])
            cx = int((x1_box + x2_box) / 2)
            cy = int((y1_box + y2_box) / 2)

            # 객체 정보 박스 그리기 (전체 표시)
            cv2.rectangle(current_frame, (x1_box, y1_box), (x2_box, y2_box), (255, 255, 0), 2)
            cv2.putText(current_frame, label, (x1_box, y1_box - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # 위험구역 내 판단
            if zone_x1 <= cx <= zone_x2 and zone_y1 <= cy <= zone_y2:
                if label in ['person', 'motorbike']:
                    person_motorbike_in_zone = True
                elif label in ['truck', 'trailer', 'bus', 'construction vehicle', 'car']:
                    construction_vehicle_in_zone = True

        # 위험 침입 + 건설차량 없음일 때만 저장 & 빨간 사각형 표시
        if person_motorbike_in_zone and not construction_vehicle_in_zone:
            cv2.rectangle(current_frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (0, 0, 255), 2)

            current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            time_str = str(timedelta(milliseconds=current_time_ms)).split('.')[0]

            cv2.putText(current_frame, "Intrusion Detected", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(current_frame, f'Time: {time_str}', (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            filename = f'filtered_frames/frame_{save_count}_{time_str.replace(":", "-")}.jpg'
            cv2.imwrite(filename, current_frame)
            print(f"[Alert] Intrusion detected - photo saved: {filename}")
            save_count += 1

            # 실시간 경고 표시 여부 설정
            show_alert_now = True
        else:
            show_alert_now = False

    # 저장된 경우에만 실시간 경고문구 표시
    if show_alert_now:
        cv2.putText(current_frame, "Warning! Intruder detected", (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    cv2.imshow("Live Detection", current_frame)  # 화면 보여주기

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 누르면 종료
        break

    frame_count += 1  # 다음 프레임

cap.release()
cv2.destroyAllWindows()
print("모니터링이 종료되었습니다.")
