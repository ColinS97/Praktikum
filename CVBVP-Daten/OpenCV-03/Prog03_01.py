import cv2 as cv
import numpy as np

image = cv.imread('CVBVP-Daten/OpenCV-03/Img03a.jpg')
if not image.data:
    print("Error loading image")
    exit

## Aufgabe 1

#cv.imshow("Image:", image)
#cv.imshow("Image mit getauschten Farbkanaelen:", cv.cvtColor(image, cv.COLOR_BGR2RGB))

height = image.shape[0]
width = image.shape[1]

#https://pyimagesearch.com/2021/01/23/splitting-and-merging-channels-with-opencv/
(B, G, R) = cv.split(image)

grayscaleimage = np.empty((height, width))
fullgrayscaleimage = image.copy()

for y in range(height):
    for x in range (width):
        pixel = image[y][x]
        grayscalevalue = int((pixel[0]+pixel[1]+pixel[2])/3)
        fullgrayscaleimage[y][x][0] = grayscalevalue
        fullgrayscaleimage[y][x][1] = grayscalevalue
        fullgrayscaleimage[y][x][2] = grayscalevalue
        grayscaleimage[y][x] = grayscalevalue

cv.imshow("Image:", fullgrayscaleimage)
cv.imshow("Image2:", grayscaleimage)
cv.imshow("Image3:", cv.cvtColor(image, cv.COLOR_BGR2GRAY))
if cv.waitKey(0) == ord('q'):
        exit()
cv.destroyAllWindows()
cv.imwrite('CVBVP-Daten/OpenCV-03/out/blue.jpg', B)
cv.imwrite('CVBVP-Daten/OpenCV-03/out/green.tif', G)
cv.imwrite('CVBVP-Daten/OpenCV-03/out/red.png', R)
cv.imwrite('CVBVP-Daten/OpenCV-03/out/gray.png', fullgrayscaleimage)
cv.imwrite('CVBVP-Daten/OpenCV-03/out/convertedcolorspace.png', cv.cvtColor(image, cv.COLOR_BGR2RGB))

## Aufgabe 2