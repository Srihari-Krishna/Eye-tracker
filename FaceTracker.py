import cv2 as cv
import mediapipe as mp
import time
import pyautogui as pg

capture = cv.VideoCapture(0)
pTime = cTime = 0
mpFace = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pg.size()
#pg.FAILSAFE = False

while True:
    isTrue, img = capture.read()
    img = cv.flip(img, 1)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    output = mpFace.process(imgRGB)

    if output.multi_face_landmarks:
        for landmarks in output.multi_face_landmarks:
            for id, landmark in enumerate(landmarks.landmark[474:478]):
                x = int(landmark.x * img.shape[1])
                y = int(landmark.y * img.shape[0])
                cv.circle(img, (x, y), 3, (0, 255, 0), 1)
                if id == 1:
                    screen_x = int(landmark.x * screen_w)
                    screen_y = int(landmark.y * screen_h)
                    pg.moveTo(screen_x, screen_y)

            left = [landmarks.landmark[145], landmarks.landmark[159]]
            for landmark in left:
                x = int(landmark.x * img.shape[1])
                y = int(landmark.y * img.shape[0])
                cv.circle(img, (x, y), 3, (0, 255, 255), 1)

            if(left[0].y - left[1].y) < 0.015:
                pg.click()
                pg.sleep(1)

    cv.imshow("Video", img)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv.waitKey(1)
