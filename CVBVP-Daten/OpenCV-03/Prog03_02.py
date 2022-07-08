import cv2 as cv
import numpy as np

cap = cv.VideoCapture('CVBVP-Daten/OpenCV-03/Vid03a.mov')

if not cap.isOpened():
    print("Cannot open video file.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    #to display both in the same frame
    channel3gray = cv.merge([gray, gray, gray])
    numpy_horizontal_concat = np.concatenate((frame, channel3gray), axis=1)

    cv.imshow('frame', numpy_horizontal_concat)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()