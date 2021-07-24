import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
import darknet


# Load image, grayscale, Otsu's threshold
image = cv2.imread('img.png')
lower_color = np.array([190, 20, 190])
upper_color = np.array([255, 55, 255])
mask = cv2.inRange(image, lower_color, upper_color)
result = cv2.bitwise_and(image, image, mask=mask)

gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
ret, tresh = cv2.threshold(gray, 255, 255, cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(
    gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]


for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if(h > 100 and w > 100):
        cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)

cv2.imshow('image', image)
cv2.waitKey()
