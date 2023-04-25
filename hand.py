import cv2
import mediapipe as mp
import numpy as np
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands

hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
while True:
    success, image = cap.read()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    blackImage = np.zeros(np.shape(imageRGB),dtype=np.uint8)
    results = hands.process(imageRGB)
    if results.multi_hand_landmarks:
        # print(list(results.multi_hand_landmarks))
        for handLms in results.multi_hand_landmarks: # working with each hand
            print(len(handLms.landmark))
            for id, lm in enumerate(handLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # if id == 20 :
                #     cv2.circle(blackImage, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(blackImage, handLms, mpHands.HAND_CONNECTIONS)
    cv2.imshow("Output", blackImage)
    cv2.waitKey(1)
    