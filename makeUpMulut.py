import cv2
import numpy as np
import dlib


cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

r = 0
g = 180
b = 255


def createBoxLips(img, points, scale=5, masked=False, cropped=True):
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
    print(myPoints[48:61])
    imgLips = createBoxLips(img, myPoints[48:61], 2, masked=True, cropped=False)

    imgColorLips = np.zeros_like(imgLips)
    imgColorLips[:] = b, g, r
    imgColorLips = cv2.bitwise_and(imgLips, imgColorLips)
    imgColorLips = cv2.GaussianBlur(imgColorLips, (7, 7), 10)
    imgColorLips = cv2.addWeighted(imgOriginal, 1, imgColorLips, 0.4, 0)
    cv2.imshow('BGR', imgColorLips)


# cv2.imshow('Original', imgOriginal)
cv2.imwrite('hasilEdit.jpg', imgColorLips)
cv2.waitKey(0)
cv2.destroyAllWindows()
