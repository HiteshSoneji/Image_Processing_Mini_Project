import sys
import numpy as np
import cv2

MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

def main():
    kernel = np.ones((5, 5), np.uint8)
    imgTraining = cv2.imread("Training_Datasets/Hitesh_training1.jpg")
    imgTraining1 = cv2.imread("Training_Datasets/Hitesh_training.jpg")

    imgTraining2 = cv2.imread("Training_Datasets/Soman_training.jpg")
    imgTraining3 = cv2.imread("Training_Datasets/Soman_training1.jpg")

    imgTraining4 = cv2.imread("Training_Datasets/Vyas_training.jpg")
    imgTraining5 = cv2.imread("Training_Datasets/Vyas_training1.jpg")

    imgTraining = cv2.resize(imgTraining, (1516, 800))
    imgTraining1 = cv2.resize(imgTraining1, (1516, 800))

    imgTraining2 = cv2.resize(imgTraining2, (1516, 800))
    imgTraining3 = cv2.resize(imgTraining3, (1516, 800))

    imgTraining4 = cv2.resize(imgTraining4, (1516, 800))
    imgTraining5 = cv2.resize(imgTraining5, (1516, 800))

    imgGray = cv2.cvtColor(imgTraining, cv2.COLOR_BGR2GRAY)
    # imgGray = cv2.erode(imgGray, kernel, iterations=2)

    imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)

    imgGray1 = cv2.cvtColor(imgTraining1, cv2.COLOR_BGR2GRAY)
    # imgGray1 = cv2.erode(imgGray1, kernel, iterations=2)

    imgBlurred1 = cv2.GaussianBlur(imgGray1, (5, 5), 0)

    imgGray2 = cv2.cvtColor(imgTraining2, cv2.COLOR_BGR2GRAY)
    imgBlurred2 = cv2.GaussianBlur(imgGray2, (5, 5), 0)

    imgGray3 = cv2.cvtColor(imgTraining3, cv2.COLOR_BGR2GRAY)
    imgBlurred3 = cv2.GaussianBlur(imgGray3, (5, 5), 0)

    imgGray4 = cv2.cvtColor(imgTraining4, cv2.COLOR_BGR2GRAY)
    imgBlurred4 = cv2.GaussianBlur(imgGray4, (5, 5), 0)

    imgGray5 = cv2.cvtColor(imgTraining5, cv2.COLOR_BGR2GRAY)
    imgBlurred5 = cv2.GaussianBlur(imgGray5, (5, 5), 0)

# filter image from grayscale to black and white
    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    imgThresh1 = cv2.adaptiveThreshold(imgBlurred1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    imgThresh2 = cv2.adaptiveThreshold(imgBlurred2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    imgThresh3 = cv2.adaptiveThreshold(imgBlurred3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    imgThresh4 = cv2.adaptiveThreshold(imgBlurred4, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    imgThresh5 = cv2.adaptiveThreshold(imgBlurred5, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    imgThreshCopy = imgThresh.copy()
    imgThreshCopy1 = imgThresh1.copy()

    imgThreshCopy2 = imgThresh2.copy()
    imgThreshCopy3 = imgThresh3.copy()

    imgThreshCopy4 = imgThresh4.copy()
    imgThreshCopy5 = imgThresh5.copy()

    npaContours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # print(npaContours)
    # print(npaHierarchy)

    npaContours1, npaHierarchy1 = cv2.findContours(imgThreshCopy1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    npaContours2, npaHierarchy2 = cv2.findContours(imgThreshCopy2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    npaContours3, npaHierarchy3 = cv2.findContours(imgThreshCopy3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    npaContours4, npaHierarchy4 = cv2.findContours(imgThreshCopy4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    npaContours5, npaHierarchy5 = cv2.findContours(imgThreshCopy5, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    npaFlattenedImages = np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

    intClassifications = []

    intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                     ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                     ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z'), ord('a'), ord('b'), ord('c'), ord('d'),
                     ord('e'), ord('f'), ord('g'), ord('h'), ord('i'), ord('j'), ord('k'), ord('l'), ord('m'), ord('n'),
                     ord('o'), ord('p'), ord('q'), ord('r'), ord('s'), ord('t'), ord('u'), ord('v'), ord('w'), ord('x'),
                     ord('y'), ord('z'), ord('.'), ord(','), ord('?'), ord('!'), ord(' ')]

    for npaContour in npaContours:
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)

            # draw rectangle around each contour as we ask user for input
            cv2.rectangle(imgTraining, (intX, intY), (intX + intW, intY + intH), (0, 0, 255), 2)

            imgROI = imgThresh[intY:intY + intH, intX:intX + intW]
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

            cv2.imshow("imgROI", imgROI)
            cv2.imshow("imgROIResized", imgROIResized)
            cv2.imshow("training_numbers.png", imgTraining)

            intChar = cv2.waitKey(0)
            print(intChar)

            if intChar == 27:
                sys.exit()
            elif intChar in intValidChars:

                intClassifications.append(intChar)
                npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)

    for npaContour in npaContours1:
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)

            cv2.rectangle(imgTraining1, (intX, intY), (intX + intW, intY + intH), (0, 0, 255), 2)

            imgROI = imgThresh1[intY:intY + intH, intX:intX + intW]
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

            cv2.imshow("imgROI", imgROI)
            cv2.imshow("imgROIResized", imgROIResized)
            cv2.imshow("training_numbers_small.png", imgTraining1)

            intChar = cv2.waitKey(0)
            print(intChar)

            if intChar == 27:
                sys.exit()
            elif intChar in intValidChars:

                intClassifications.append(intChar)
                npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)

    for npaContour in npaContours2:
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)

            cv2.rectangle(imgTraining2, (intX, intY), (intX + intW, intY + intH), (0, 0, 255), 2)

            imgROI = imgThresh2[intY:intY + intH, intX:intX + intW]
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

            cv2.imshow("imgROI", imgROI)
            cv2.imshow("imgROIResized", imgROIResized)
            cv2.imshow("training_numbers_small.png", imgTraining2)

            intChar = cv2.waitKey(0)
            print(intChar)

            if intChar == 27:
                sys.exit()
            elif intChar in intValidChars:

                intClassifications.append(intChar)
                npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)

    for npaContour in npaContours3:
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)

            cv2.rectangle(imgTraining3, (intX, intY), (intX + intW, intY + intH), (0, 0, 255), 2)

            imgROI = imgThresh3[intY:intY + intH, intX:intX + intW]
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

            cv2.imshow("imgROI", imgROI)
            cv2.imshow("imgROIResized", imgROIResized)
            cv2.imshow("training_numbers_small.png", imgTraining3)

            intChar = cv2.waitKey(0)
            print(intChar)

            if intChar == 27:
                sys.exit()
            elif intChar in intValidChars:

                intClassifications.append(intChar)
                npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)

    for npaContour in npaContours4:
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)

            cv2.rectangle(imgTraining4, (intX, intY), (intX + intW, intY + intH), (0, 0, 255), 2)

            imgROI = imgThresh4[intY:intY + intH, intX:intX + intW]
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

            cv2.imshow("imgROI", imgROI)
            cv2.imshow("imgROIResized", imgROIResized)
            cv2.imshow("training_numbers_small.png", imgTraining4)

            intChar = cv2.waitKey(0)
            print(intChar)

            if intChar == 27:
                sys.exit()
            elif intChar in intValidChars:

                intClassifications.append(intChar)
                npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)

    for npaContour in npaContours5:
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)

            cv2.rectangle(imgTraining5, (intX, intY), (intX + intW, intY + intH), (0, 0, 255), 2)

            imgROI = imgThresh5[intY:intY + intH, intX:intX + intW]
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

            cv2.imshow("imgROI", imgROI)
            cv2.imshow("imgROIResized", imgROIResized)
            cv2.imshow("training_numbers_small.png", imgTraining5)

            intChar = cv2.waitKey(0)
            print(intChar)

            if intChar == 27:
                sys.exit()
            elif intChar in intValidChars:

                intClassifications.append(intChar)
                npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)

    fltClassifications = np.array(intClassifications, np.float32)

    npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))

    print("\n\ntraining complete !!\n")

    np.savetxt("classifications_handwritten.txt", npaClassifications)
    np.savetxt("flattened_images_handwritten.txt", npaFlattenedImages)

    cv2.destroyAllWindows()

    return


if __name__ == "__main__":
    main()
