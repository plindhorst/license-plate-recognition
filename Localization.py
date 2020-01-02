import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.io import imread
from skimage.filters import threshold_otsu

"""
In this file, you need to define plate_detection function.
To do:
	1. Localize the plates and crop the plates
	2. Adjust the cropped plate images
Inputs:(One)
	1. image: captured frame in CaptureFrame_Process.CaptureFrame_Process function
	type: Numpy array (imread by OpenCV package)
Outputs:(One)
	1. plate_imgs: cropped and adjusted plate images
	type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
Hints:
	1. You may need to define other functions, such as crop and adjust function
	2. You may need to define two ways for localizing plates(yellow or other colors)
"""

def plot_image(img1, img2, title1="", title2=""):
	fig = plt.figure(figsize=[15,15])
	ax1 = fig.add_subplot(121)
	ax1.imshow(img1)
	ax1.set(xticks=[], yticks=[], title=title1)

	ax2 = fig.add_subplot(122)
	ax2.imshow(img2, cmap="gray")
	ax2.set(xticks=[], yticks=[], title=title2)
	plt.show()

def plate_detection(image):
	img = image

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	blur = cv2.bilateralFilter(gray, 11, 150, 150)

	edges = cv2.Canny(blur, 20, 80)

	contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	contours = sorted(contours, key=cv2.contourArea, reverse=True)

	image_copy = img.copy()
	_ = cv2.drawContours(image_copy, contours, -1, (255, 0, 255), 3)

	plate = None
	for c in contours:
		perimeter = cv2.arcLength(c, True)
		edges_count = cv2.approxPolyDP(c, 0.02 * perimeter, True)
		if len(edges_count) == 4:
			x, y, w, h = cv2.boundingRect(c)
			if (w / h < 6) & (w / h > 4):
				plate = img[y:y + h, x:x + w]
				break

	plot_image(image_copy, plate, "Original", "Affected")
	return plate


if __name__ == '__main__':
	img = cv2.imread('localization_trainingset/loc2.jpg')
	plate_detection(img)





