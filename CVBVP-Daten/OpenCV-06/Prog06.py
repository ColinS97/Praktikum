import cv2 as cv
import numpy as np

cap = cv.VideoCapture("CVBVP-Daten/OpenCV-06/Vid06a.avi")

if not cap.isOpened():
    print("Cannot open video file.")
    exit()

lastFrameBlur = None

while True:
    # Aufgabe 1
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    kernel = 5

    if lastFrameBlur is None:
        lastFrameBlur = np.empty(frame.shape, dtype=frame.dtype)
        lastFrameMedianBlur = np.empty(frame.shape, dtype=frame.dtype)
        lastFrameEqHist = np.empty(frame.shape, dtype=frame.dtype)
        lastFrame = np.empty(frame.shape, dtype=frame.dtype)

    modFrameBlur = cv.blur(frame, (5, 5))
    modFrameMedianBlur = cv.medianBlur(frame, 5)
    modFrameEqHist = cv.equalizeHist(frame)

    diffFrame = cv.absdiff(frame, lastFrame)
    diffFrameBlur = cv.absdiff(modFrameBlur, lastFrameBlur)
    diffFrameMedianBlur = cv.absdiff(modFrameMedianBlur, lastFrameMedianBlur)
    diffFrameEqHist = cv.absdiff(modFrameEqHist, lastFrameEqHist)

    frames = np.concatenate(
        (frame, modFrameBlur, modFrameMedianBlur, modFrameEqHist), axis=1
    )
    diffFrames = np.concatenate(
        (diffFrame, diffFrameBlur, diffFrameMedianBlur, diffFrameEqHist), axis=1
    )
    lastFrames = np.concatenate(
        (lastFrame, lastFrameBlur, lastFrameMedianBlur, lastFrameEqHist), axis=1
    )
    fullFrame = np.concatenate(
        (
            frames,
            lastFrames,
            diffFrames,
        ),
        axis=0,
    )
    cv.imshow("frame", fullFrame)
    if cv.waitKey(1) == ord("q"):
        break

    lastFrame = frame
    lastFrameBlur = modFrameBlur
    lastFrameMedianBlur = modFrameMedianBlur
    lastFrameEqHist = modFrameEqHist
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
