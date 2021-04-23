import pyvirtualcam
import cv2

WIDTH = 1280
HEIGHT = 720
FPS = 30
MASK_COLOR = (30, 30, 30)

# YOUR_NAME = 'Theo'
# FONT = cv2.FONT_HERSHEY_SIMPLEX
# FONT_SCALE = 1
# FONT_COLOR = (240, 240, 240)
# FONT_THICKNESS = 2

blur_kernel_width = (WIDTH // 7) | 1
blur_kernel_height = (HEIGHT // 7) | 1

in_cam = cv2.VideoCapture(0)
in_cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
in_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

with pyvirtualcam.Camera(width=WIDTH, height=HEIGHT, fps=FPS) as out_cam:
    while True:
        _, image = in_cam.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
        face_data = face_detect.detectMultiScale(image, 1.1, 3)

        for (x, y, w, h) in face_data:
            # Show ellipse on faces
            center_x = x + w//2
            center_y = y + h//2 - round(h/2 * 0.25)
            size_x = w//2 - round(w/2 * 0.1)
            size_y = h//2 + round(h/2 * 0.25)
            cv2.ellipse(image, (center_x, center_y), (size_x, size_y), 0, 0, 360, MASK_COLOR, -1)

            # Blur faces with rectangle
            # roi = image[y:y+h, x:x+w]
            # roi = cv2.GaussianBlur(roi, (blur_kernel_width, blur_kernel_height), 0)
            # image[y:y+roi.shape[0], x:x+roi.shape[1]] = roi

            # Show your name
            # image = cv2.putText(image, YOUR_NAME, (center_x, center_y), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        out_cam.send(image)
        out_cam.sleep_until_next_frame()