import cv2
import mediapipe as mp

def overlay(image, x, y, w, h, overlay_image):
    alpha = overlay_image[:, :, 3]
    mask_image = alpha / 255

    for c in range(0, 3):
        image[y-h:y+h, x-w:x+w, c] = (overlay_image[:, :, c] * mask_image) + (image[y-h:y+h, x-w:x+w, c] * (1 - mask_image))

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture('video/face_video.mp4')

image_right_eye_cat = cv2.imread('images/cat/right.png', cv2.IMREAD_UNCHANGED)
image_left_eye_cat = cv2.imread('images/cat/left.png', cv2.IMREAD_UNCHANGED)
image_nose_cat = cv2.imread('images/cat/nose.png', cv2.IMREAD_UNCHANGED)

image_right_eye_panda = cv2.imread('images/panda/right.png', cv2.IMREAD_UNCHANGED)
image_left_eye_panda = cv2.imread('images/panda/left.png', cv2.IMREAD_UNCHANGED)
image_nose_panda = cv2.imread('images/panda/nose.png', cv2.IMREAD_UNCHANGED)

image_right_eye_rabbit = cv2.imread('images/rabbit/right.png', cv2.IMREAD_UNCHANGED)
image_left_eye_rabbit = cv2.imread('images/rabbit/left.png', cv2.IMREAD_UNCHANGED)
image_nose_rabbit = cv2.imread('images/rabbit/nose.png', cv2.IMREAD_UNCHANGED)

animal_overlay = 'cat' 

with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.9) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detections:
            for detection in results.detections:
                keypoints = detection.location_data.relative_keypoints
                right_eye = keypoints[0]
                left_eye = keypoints[1]
                nose_tip = keypoints[2]

                h, w, _ = image.shape
                right_eye = (int(right_eye.x * w) - 20, int(right_eye.y * h) - 100)
                left_eye = (int(left_eye.x * w) + 20, int(left_eye.y * h) - 100)
                nose_tip = (int(nose_tip.x * w), int(nose_tip.y * h))

                if animal_overlay == 'cat':
                    overlay(image, *right_eye, 50, 50, image_right_eye_cat)
                    overlay(image, *left_eye, 50, 50, image_left_eye_cat)
                    overlay(image, *nose_tip, 150, 50, image_nose_cat)
                elif animal_overlay == 'dog':
                    overlay(image, *right_eye, 50, 50, image_right_eye_panda)
                    overlay(image, *left_eye, 50, 50, image_left_eye_panda)
                    overlay(image, *nose_tip, 150, 50, image_nose_panda)

                elif animal_overlay == 'rabbit':
                    overlay(image, *right_eye, 50, 50, image_right_eye_rabbit)
                    overlay(image, *left_eye, 50, 50, image_left_eye_rabbit)
                    overlay(image, *nose_tip, 150, 50, image_nose_rabbit)

        cv2.imshow('MediaPipe Face Detection', cv2.resize(image, None, fx=0.7, fy=0.7))

        key = cv2.waitKey(20)
        if key == ord('q'):
            break
        elif key == ord('c'):
            animal_overlay = 'cat'
        elif key == ord('p'):
            animal_overlay = 'dog'
        elif key == ord('r'):
            animal_overlay = 'rabbit'

cap.release()
cv2.destroyAllWindows()
