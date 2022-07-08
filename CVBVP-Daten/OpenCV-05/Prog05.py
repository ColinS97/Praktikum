import cv2 as cv
import numpy as np

# 5.1
## erstellen sie ein grauwertbild
g_img = cv.imread("CVBVP-Daten/OpenCV-05/Img05a.jpg", cv.IMREAD_GRAYSCALE)
img = cv.imread("CVBVP-Daten/OpenCV-05/Img05a.jpg")
cv.imwrite("CVBVP-Daten/OpenCV-05/out/Img05a_grayscale.jpg", g_img)
## erstellen sie ein histogramm aus dem grauwertbild
# https://docs.opencv.org/3.4/d8/dbc/tutorial_histogram_calculation.html
# https://docs.opencv.org/4.x/d6/dc7/group__imgproc__hist.html#ga4b2b5fd75503ff9e6844cc4dcdaed35d
histSize = 256
histRange = (0, 256)  # the upper boundary is exclusive
accumulate = False
g_hist = cv.calcHist(g_img, [0], None, [histSize], histRange, accumulate=accumulate)

hist_w = 512
hist_h = 400
bin_w = int(round(hist_w / histSize))
histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

cv.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)

for i in range(1, histSize):
    cv.line(
        histImage,
        (bin_w * (i - 1), hist_h - int(g_hist[i - 1])),
        (bin_w * (i), hist_h - int(g_hist[i])),
        (255, 0, 0),
        thickness=2,
    )
cv.imwrite("CVBVP-Daten/OpenCV-05/out/Img05a_grayscale_hist.jpg", histImage)

## erstelle gauß pyramide des bildes
# https://docs.opencv.org/3.4/d4/d1f/tutorial_pyramids.html
rows, cols, _channels = map(int, img.shape)
## pyrup = downsampling = gaussian
gauss_pyr = cv.pyrUp(img, dstsize=(2 * cols, 2 * rows))
cv.imwrite("CVBVP-Daten/OpenCV-05/out/Img05a_gauss_pyr.jpg", gauss_pyr)

## pyrdown = upsampling = laplacian

laplace_pyr = cv.pyrDown(img, dstsize=(cols // 2, rows // 2))
cv.imwrite("CVBVP-Daten/OpenCV-05/out/Img05a_laplace_pyr.jpg", laplace_pyr)


##5.2

# führen sie die fourier transformation und die rücktransformation auf das grauwertbild aus
# https://docs.opencv.org/4.x/de/dbc/tutorial_py_fourier_transform.html


m = cv.getOptimalDFTSize(g_img.shape[0])
n = cv.getOptimalDFTSize(g_img.shape[1])
padded = cv.copyMakeBorder(
    g_img, 0, m - g_img.shape[0], 0, n - g_img.shape[1], cv.BORDER_CONSTANT
)

planes = [np.float32(padded), np.zeros(padded.shape, dtype=np.float32)]
complexImg = cv.merge(planes, 2)


complexImg = cv.dft(complexImg)

cv.split(complexImg, planes)
planes[0] = cv.magnitude(planes[0], planes[1])

mag = planes[0]
# mag += cv.Scalar.all(1)

mag = cv.log(mag)
mag = mag(cv.rectangle(0, 0, mag.shape[1] - 2, mag.shape[0] - 2))
cx = mag.cols // 2
cy = mag.rows // 2
tmp = cv.Mat()

q0 = cv.Mat(mag, cv.rectangle(0, 0, cx, cy))
q1 = cv.Mat(mag, cv.rectangle(cx, 0, cx, cy))
q2 = cv.Mat(mag, cv.rectangle(0, cy, cx, cy))
q3 = cv.Mat(mag, cv.rectangle(cx, cy, cx, cy))

q0.copyTo(tmp)
q3.copyTo(q0)
tmp.copyTo(q3)
q1.copyTo(tmp)
q2.copyTo(q1)
tmp.copyTo(q2)
cv.normalize(mag, mag, 0, 1, cv.NORM_MINMAX)

cv.imwrite("CVBVP-Daten/OpenCV-05/out/Img05a_magnitude_spectrum.jpg", mag)
dft = mag

retransform = cv.idft(dft, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
cv.imwrite("CVBVP-Daten/OpenCV-05/out/Img05a_reverse_fourier.jpg", retransform)

# zerlege bild in real und imaginärteil

real, imaginary = cv.split(dft)

print(real.shape)
print(imaginary.shape)

# zeige bild im frequenz und im ortsraum


magnitude_spectrum = 20 * np.log(cv.magnitude(real, imaginary))
cv.imwrite(
    "CVBVP-Daten/OpenCV-05/out/Img05a_frequency_magnitude.jpg", magnitude_spectrum
)

# verändere amplitude und phase aller pixel

# setze amplitude und phase an manchen stellen auf 0

# leeres bild erzeugen und einige werte ungleich 0

# ergebnisse in frequenz und ortsraum
