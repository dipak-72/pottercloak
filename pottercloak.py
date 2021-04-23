import cv2
import numpy as np
import time

print("Get ready to get invisible!!")
print("in...")
print("3")
print("2")
print("1")

capture = cv2.VideoCapture(0)
time.sleep(3)

for i in range(60):
    ret, background = capture.read()

while (capture.isOpened()):
    ret, image = capture.read()

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    value = (5, 5)
    blur = cv2.GaussianBlur(hsv, value, 0)
    
    lower_r = np.array([0, 100, 100])
    upper_r = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_r, upper_r)

    part1 = cv2.bitwise_and(background, background, mask = mask)

    mask = cv2.bitwise_not(mask)

    part2 = cv2.bitwise_and(image, image, mask = mask)

    mask = part1 + part2
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

    cv2.imshow('cloak', mask)

    if cv2.waitKey(5) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()