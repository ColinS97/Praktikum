import cv2 as cv
image = cv.imread('CVBVP-Daten/OpenCV-01/Img01a.jpg')
if not image.data:
    print("Error loading image")
    exit
cv.namedWindow("Image:")
cv.imshow("Image:", image)
cv.waitKey(0) 
cv.destroyAllWindows('Image:')
#plt.imshow(image)
#plt.show()
