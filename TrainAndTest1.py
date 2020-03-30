import cv2
import numpy as np
import operator

MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

class ContourWithData():

    npaContour = None
    boundingRect = None
    intRectX = 0
    intRectY = 0
    intRectWidth = 0
    intRectHeight = 0
    fltArea = 0.0

    def calculateRectTopLeftPointAndWidthAndHeight(self):
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):
        if self.fltArea < MIN_CONTOUR_AREA: return False
        return True

def main():
    allContoursWithData = []
    validContoursWithData = []

    npaClassifications = np.loadtxt("Text_Files/dataset_handwritten.txt", np.float32)
    npaFlattenedImages = np.loadtxt("Text_Files/dataset_handwritten_1D.txt", np.float32)


# try:
#     except:
#         print ("error, unable to open dataset.txt, exiting program\n")
#         os.system("pause")
#         return
#
#     try:
#     except:
#         print("error, unable to open dataset_1D.txt, exiting program\n")
#         os.system("pause")
#         return


    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))

    kNearest = cv2.ml.KNearest_create()

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    kernel = np.ones((2,2), np.uint8)

    imgTestingNumbers = cv2.imread("Testing_Images/hand1.jpg")

    imgTestingNumbers = cv2.erode(imgTestingNumbers, kernel, iterations=1)

    imgTestingNumbers = cv2.resize(imgTestingNumbers, (500, 252))


    imgGray = cv2.cvtColor(imgTestingNumbers, cv2.COLOR_BGR2GRAY)

    #Add Skew Here
    bit = cv2.bitwise_not(imgGray)

    thresh = cv2.threshold(bit, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    # elif angle==0:
    #     angle=angle
    elif angle>-45:
        angle = (-angle - 90)
    else:
        angle = -angle

    (h, w) = imgTestingNumbers.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(imgTestingNumbers, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    print("[INFO] angle: {:.3f}".format(angle))
    cv2.imshow("Rotated", rotated)

    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)

    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    imgThreshCopy = imgThresh.copy()



    npaContours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for npaContour in npaContours:
        contourWithData = ContourWithData()
        contourWithData.npaContour = npaContour
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight()
        contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)
        allContoursWithData.append(contourWithData)

    for contourWithData in allContoursWithData:
        if contourWithData.checkIfContourIsValid():
            validContoursWithData.append(contourWithData)

    validContoursWithData.sort(key = operator.attrgetter("intRectX"))

    strFinalString = ""
    strFinalString1 = []

    for contourWithData in validContoursWithData:

        cv2.rectangle(imgTestingNumbers, (contourWithData.intRectX, contourWithData.intRectY),
                      (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),      # lower right corner
                      (0, 255, 0), 2)

        imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,
                 contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]

        imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

        npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

        npaROIResized = np.float32(npaROIResized)

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)
        print(npaResults)

        # matches = result == test_labels
        # correct = np.count_nonzero(matches)
        # accuracy = correct * 100.0 / result.size

        # print("Accuracy is = %.2f" %accuracy +"%")

        strCurrentChar = str(chr(int(npaResults[0][0])))

        strFinalString = strFinalString + strCurrentChar
        strFinalString1.append(strCurrentChar)

    print("\n" + strFinalString + "\n")
    print(sorted(strFinalString1))

    cv2.imshow("imgTestingNumbers", imgTestingNumbers)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    return

if __name__ == "__main__":
    main()









