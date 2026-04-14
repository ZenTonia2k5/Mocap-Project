import cv2
import torch
from mmpose.apis import MMPoseInferencer
import time

# Trước vòng lặp while
pTime = 0

# 1. Khởi tạo model (Dùng 'human' cho nhẹ và mượt)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
inferencer = MMPoseInferencer(pose2d='vitpose-b', device=device)

# 2. Mở Camera stream
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Tạo một cửa sổ OpenCV cố định để stream video
cv2.namedWindow('Pose Stream', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Pose Stream', 1280, 720)

print("--- ĐANG STREAM VIDEO TỪ CAMERA ---")
print("Nhấn 'q' để thoát")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 3. Dự đoán nhưng KHÔNG dùng show=True của inferencer
    # Dùng return_vis=True để lấy ảnh đã vẽ về tự hiển thị
    result_it = inferencer(inputs=frame, return_vis=True)
    result = next(result_it)
    
    # Lấy tấm ảnh đã được vẽ khung xương từ AI
    # result['visualization'] là một danh sách các ảnh (thường là 1)
    viz_frame = result['visualization'][0]
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(viz_frame, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    # 4. Hiển thị vào cửa sổ stream duy nhất
    viz_frame_correct_color = cv2.cvtColor(viz_frame, cv2.COLOR_RGB2BGR)

    cv2.imshow('Pose Stream', viz_frame_correct_color)


    # Thoát nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()