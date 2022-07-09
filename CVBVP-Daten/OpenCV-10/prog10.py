import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


# img10a01 und 02
# in der vl gibt es ein besipielprogramm
# https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html

imgL = cv.imread("CVBVP-Daten/OpenCV-10/img10a01.png", cv.IMREAD_GRAYSCALE)
imgR = cv.imread("CVBVP-Daten/OpenCV-10/img10a02.png", cv.IMREAD_GRAYSCALE)

stereo = cv.StereoBM_create(numDisparities=32, blockSize=15)
disparity = stereo.compute(imgL, imgR)
normalized = cv.normalize(
    disparity, disparity, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U
)
normalized.convertTo(normalized, cv.CV_8UC1)
# get fundamentalmat from 8 selection points in either image
# computeCorrespondEpilines
# drawEpilines
# stereoRectifyUncalibrated(selPoints1, selPoints2, Fundamentalmatrix, imgSize.size(), OutputHomography1H1, OututHomoGraphyH2)
# ich glaube reprojectTo3D ist nicht moeglich da man dafuer die kamera kalibrieren muesste und das ohne ein schachbrett glaube nicht geht
# warpTransform mit den Ergebnissen aus recitify uncalibrated
# epipolarlinien sind
# StereoBM.create.compute(warpTransforemdImage1, warpTransformedImage2, disparity)
# das ergibt dann das 3D bild


plt.imshow(disparity, "gray")
plt.show()
