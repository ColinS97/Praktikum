import numpy as np
import cv2 as cv

cap = cv.VideoCapture(cv.samples.findFile("CVBVP-Daten/OpenCV-11/Vid11d.avi"))
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

kalman = cv.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)

kalman.transitionMatrix = np.array(
    [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
)

kalman.processNoiseCov = (
    np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    * 0.03
)

measurement = np.array((2, 1), np.float32)
prediction = np.zeros((2, 1), np.float32)

while 1:
    ret, frame2 = cap.read()
    if not ret:
        print("No frames grabbed!")
        break
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    diff = cv.absdiff(prvs, next)
    # get minmaxmloc
    minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(diff)
    npmaxLoc = np.array(maxLoc, dtype=np.float32)
    kalman.correct(npmaxLoc)
    prediction = kalman.predict()

    cv.circle(prvs, maxLoc, 5, (255, 0, 0), 2)
    cv.circle(prvs, (int(prediction[0]), int(prediction[1])), 5, (0, 255, 0), 2)
    cv.imshow("diff", diff)
    cv.imshow("prvs", cv.cvtColor(prvs, cv.COLOR_GRAY2BGR))
    prvs = next
    # end loop when key q is pressed
    k = cv.waitKey(30) & 0xFF
    if k == ord("q"):
        break

# https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
