import cv2
import numpy as np
import dlib


cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

r = 0
g = 180
b = 255


def createBoxEyes(img, points, scale=5, masked=False, cropped=True):
    if masked:
        mask = np.zeros_like(img)
        #block
        mask = cv2.fillPoly(mask, [points], (255, 255, 255))
        #line
        # mask = cv2.polylines(mask, [points], False, (255, 255, 255), thickness=10)
        # img = cv2.bitwise_and(img, mask)
        # cv2.imshow('Mask', img)

    if cropped:
        bbox = cv2.boundingRect(points)
        x, y, w, h = bbox
        imgCrop = img[y:y + h, x:x + w]
        imgCrop = cv2.resize(imgCrop, (0, 0), None, scale, scale)
        return imgCrop

    else:
        return mask


# foto
img = cv2.imread('woman.jpg')
img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
imgOriginal = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detector(imgGray)

for face in faces:
    x1, y1 = face.left(), face.top()
    x2, y2 = face.right(), face.bottom()
    # tampilan bounding box
    # imgOriginal = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    landmarks = predictor(imgGray, face)
    myPoints = []
    for n in range(68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        myPoints.append([x, y])

    myPoints = np.array(myPoints)
    # print(myPoints[48:61])
    imgEyeRight = createBoxEyes(img, myPoints[36:42], 2, masked=True, cropped=False)
    imgEyeLeft = createBoxEyes(img, myPoints[42:47], 2, masked=True, cropped=False)

    imgColorEyeRight = np.zeros_like(imgEyeRight)
    imgColorEyeRight[:] = b, g, r
    imgColorEyeRight = cv2.bitwise_and(imgEyeRight, imgColorEyeRight)
    imgColorEyeRight = cv2.GaussianBlur(imgColorEyeRight, (7, 7), 10)

    imgColorEyeLeft = np.zeros_like(imgEyeLeft)
    imgColorEyeLeft[:] = b, g, r
    imgColorEyeLeft = cv2.bitwise_and(imgEyeLeft, imgColorEyeLeft)
    imgColorEyeLeft = cv2.GaussianBlur(imgColorEyeLeft, (7, 7), 10)

    imgColorEyes = cv2.addWeighted(imgOriginal, 1, imgColorEyeRight, 0.4, 0)
    imgColorEyes = cv2.addWeighted(imgColorEyes, 1, imgColorEyeLeft, 0.4, 0)
    cv2.imshow('Mata', imgColorEyes)


# cv2.imshow('Original', imgOriginal)
cv2.imwrite('hasilEdit.jpg', imgColorEyes)
cv2.waitKey(0)
cv2.destroyAllWindows()
