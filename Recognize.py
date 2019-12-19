import cv2
import numpy as np
import os
import glob

"""
In this file, you will define your own segment_and_recognize function.
To do:
	1. Segment the plates character by character
	2. Compute the distances between character images and reference character images(in the folder of 'SameSizeLetters' and 'SameSizeNumbers')
	3. Recognize the character by comparing the distances
Inputs:(One)
	1. plate_imgs: cropped plate images by Localization.plate_detection function
	type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
Outputs:(One)
	1. recognized_plates: recognized plate characters
	type: list, each element in recognized_plates is a list of string(Hints: the element may be None type)
Hints:
	You may need to define other functions.
"""


def brightness(im, n):
    return np.where((255 - im) < n, 255, im + n)


def get_same_pixels(im_1, im_2):
    same = 0
    for i in range(len(im_1)):
        for j in range(len(im_1[i])):
            if im_1[i][j] == im_2[i][j]:
                same += 1
    return same


def ocr(im):
    height, width = im.shape
    n = height * width

    letters = []
    values = []
    im_letters_list = []
    for file in glob.glob("Characters/*.png"):
        im_letter = cv2.imread(file)
        im_letter = cv2.cvtColor(im_letter, cv2.COLOR_BGR2GRAY)
        im_letter = cv2.resize(im_letter, (width, height))

        same = get_same_pixels(im, im_letter)

        letters.append(os.path.basename(file.replace('.png', '')))
        values.append(round((same * 100) / n, 2))
        im_letters_list.append(im_letter)

    indices = np.argsort(values)

    '''
    for index in range(len(indices)):
        i = indices[index]
        print(letters[i] + " : " + str(values[i]))

    print(letters[indices[len(indices) - 1]].upper() + " : " + str(values[indices[len(indices) - 1]]))

    debug = np.concatenate((img, im_letters_list[indices[len(indices) - 1]]), axis=1)

    cv2.imshow("debug", debug)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    return letters[indices[len(indices) - 1]].upper(), values[indices[len(indices) - 1]]


def segment_and_recognize(plate_images):
    result = []
    result2 = []
    for i in range(len(plate_images)):
        im = plate_images[i]
        # make grey scale
        im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # increase contrast
        im_contrast = cv2.equalizeHist(im_grey)
        # remove noise
        im_noise = cv2.medianBlur(im_contrast, 1)
        # binary image
        ret, im_bin = cv2.threshold(im_noise, 40, 255, cv2.THRESH_BINARY)
        # make black areas bigger
        kernel = np.ones((3, 3), np.uint8)
        im_erosion = cv2.erode(im_bin, kernel, iterations=1)

        # get contours of letters
        contours, hierarchy = cv2.findContours(im_erosion.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # create bounding boxes
        bounding_boxes = []
        for j, contour in enumerate(contours):
            bounding_box = cv2.boundingRect(contour)
            if (bounding_box[0] != 0 and bounding_box[1] != 0
                    and bounding_box[2] != im_erosion.shape[1] and bounding_box[3] != im_erosion.shape[0] and hierarchy[
                        0, j, 3] == 0):
                bounding_boxes.append(bounding_box)

        # show bounding boxes
        img_boxes = im_erosion.copy()
        for bounding_box in bounding_boxes:
            x, y, w, h = bounding_box
            cv2.rectangle(img_boxes, (x, y), (x + w, y + h), (0, 0, 0), 1)

        # get only characters boxes
        img_characters = np.full([len(im_grey), len(im_grey[0])], 255, dtype=np.uint8)

        for bounding_box in bounding_boxes:
            x, y, w, h = bounding_box
            img_characters[y:y + h, x:x + w] = im_grey[y:y + h, x:x + w]

        # process characters boxes
        im_characters_contrast = cv2.equalizeHist(img_characters)
        ret, img_characters_bin = cv2.threshold(im_characters_contrast, 60, 255, cv2.THRESH_BINARY)

        # get contours of characters
        contours, hierarchy = cv2.findContours(img_characters_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # create bounding boxes for characters
        bounding_boxes = []
        for j, contour in enumerate(contours):
            bounding_box = cv2.boundingRect(contour)
            if (bounding_box[0] != 0 and bounding_box[1] != 0
                    and bounding_box[2] != img_characters_bin.shape[1] and bounding_box[3] != img_characters_bin.shape[
                        0] and hierarchy[0, j, 3] == 0):
                bounding_boxes.append(bounding_box)

        # show bounding boxes for characters
        sum_width = 0
        sum_height = 0
        img_characters_boxes = img_characters_bin.copy()
        for bounding_box in bounding_boxes:
            x, y, w, h = bounding_box
            sum_width += w
            sum_height += h
            cv2.rectangle(img_characters_boxes, (x, y), (x + w, y + h), (0, 0, 0), 1)

        images_grey = np.concatenate((im_grey, im_contrast, im_noise, im_bin, im_erosion, img_boxes,), axis=1)
        result.append(images_grey)

        characters_grey = np.concatenate(
            (img_characters, im_characters_contrast, img_characters_bin,
             img_characters_boxes), axis=1)
        result2.append(characters_grey)

        if len(bounding_boxes) == 0:
            print("Plate " + str(i) + " : ERROR")
            continue

        average_width = sum_width / len(bounding_boxes)
        average_height = sum_height / len(bounding_boxes)

        # perform ocr for each character
        license_plate = ""
        probabilities = []
        bounding_boxes.sort(key=lambda x: x[0])
        width_threshold = average_width / 2
        height_threshold = average_height / 2
        height, width = img_characters_bin.shape
        for bounding_box in bounding_boxes:
            x, y, w, h = bounding_box
            if w >= width_threshold and h >= height_threshold:
                # cv2.imwrite('TrainingSet/Characters/' + str(i) + "-" + str(characters_n) + '.png', img_characters_bin[y:y + h, x:x + w])
                letter, probability = ocr(img_characters_bin[y:y + h, x:x + w])
                license_plate += letter
                probabilities.append(probability)
            elif y - 10 <= height/2 <= y + h + 10:
                license_plate += "-"

        print("Plate " + str(i) + " : " + license_plate + " (" + str(round(np.average(probabilities), 2)) + "%)")

    debug = result[0]
    debug2 = result2[0]
    height, width = result[0].shape

    for i in range(1, len(result)):
        result[i] = cv2.resize(result[i], (width, height))
        debug = np.concatenate((debug, result[i]), axis=0)

    height, width = result2[0].shape

    for i in range(1, len(result2)):
        result2[i] = cv2.resize(result2[i], (width, height))
        debug2 = np.concatenate((debug2, result2[i]), axis=0)

    cv2.imshow("debug", debug)
    cv2.imshow("debug2", debug2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':

    images = []
    for file in glob.glob("TrainingSet/Plates/*.png"):
        images.append(cv2.imread(file))

    segment_and_recognize(images)