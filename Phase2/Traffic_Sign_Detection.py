import numpy as np
import cv2
import dippykit as dip
import imutils

img = cv2.imread('C:/Users/shubham sidhwa/Downloads/TrainIJCNN2013/png_TrainIJCNN2013/00351.png')

r = img.copy()
# set blue and green channels to 0
img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lower mask (0-10)
lower_red = np.array([17,15,100])
upper_red = np.array([50, 56, 200])
mask0 = cv2.inRange(img, lower_red, upper_red)

res_red = cv2.bitwise_and(img,img, mask = mask0)
#cv2.imshow('image',output_img)

cv2.imshow('imager',res_red)

lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])

# Threshold the HSV image to get only blue colors
mask1 = cv2.inRange(img_hsv, lower_blue, upper_blue)

# Bitwise-AND mask and original image
res_blue = cv2.bitwise_and(img,img, mask = mask1)

cv2.imshow('imageb',res_blue)

imgray = cv2.cvtColor(res_red,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
res_red = cv2.drawContours(res_red, contours, -1, (0,255,0), 3)
cv2.imshow('resimgr',res_red)

imgray = cv2.cvtColor(res_blue,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
res_blue = cv2.drawContours(res_blue, contours, -1, (0,255,0), 3)
cv2.imshow('resimgb',res_blue)
cv2.waitKey(0)
cv2.destroyAllWindows()

##for cnt in contours:
##    area = cv2.contourArea(cnt)
##    print(area)
#print(res)
##
##smoothedInput = cv2.GaussianBlur(res, (7,7), 2);
##edged = cv2.Canny(smoothedInput, 100, 200);
##edged=imutils.resize(edged,width=500,height=500);
##smoothedInput_red = cv2.GaussianBlur(res_red, (7,7), 2);
##edged_red = cv2.Canny(smoothedInput_red, 20, 30);
##edged_red=imutils.resize(edged_red,width=500,height=500);
##
##
##

