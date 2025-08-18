import cv2

drawing = False
ix, iy = -1, -1
rx1, ry1, rx2, ry2 = 0, 0, 0, 0  # 위험구역 좌표

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rx1, ry1, rx2, ry2

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp = frame.copy()
            cv2.rectangle(temp, (ix, iy), (x, y), (0, 0, 255), 2)
            cv2.imshow('Set Danger Zone', temp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rx1, ry1 = min(ix, x), min(iy, y)
        rx2, ry2 = max(ix, x), max(iy, y)
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
        cv2.imshow('Set Danger Zone', frame)
        print(f"위험구역 설정됨: ({rx1}, {ry1}) → ({rx2}, {ry2})")

# 첫 프레임으로 위험구역 설정
cap = cv2.VideoCapture('../ee.mp4')
ret, frame = cap.read()
cap.release()

if not ret:
    print("영상 로딩 실패")
    exit()

cv2.namedWindow('Set Danger Zone')
cv2.setMouseCallback('Set Danger Zone', draw_rectangle)
cv2.imshow('Set Danger Zone', frame)

print("마우스로 위험구역을 지정하세요. 창에서 's'를 누르면 저장됩니다.")
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        print("위험구역 저장 완료")
        break
    elif key == 27:
        print("취소됨")
        exit()

cv2.destroyAllWindows()