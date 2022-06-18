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
        # mask = cv2.fillPoly(mask, [points], (255, 255, 255))
        #line
        mask = cv2.polylines(mask, [points], True, (255, 255, 255), thickness=2)
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
    for n in range(64):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        if n >=2 and n<=4:
            print('~~')
            print(x)
            x += 10
            print('!!')
            print(x)
            # cv2.cvtColor()

        elif n>=12 and n<=14:
            print('~')
            print(x)
            x -= 10
            print('!')
            print(x)

        myPoints.append([x, y])
        # cv2.circle(imgOriginal, (x, y), 2, (50, 50, 255), cv2.FILLED)
        # cv2.putText(imgOriginal, str(n), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1)



    # print(myPoints[48:61])

    myPoints = np.array(myPoints)
    print(myPoints[12:15])
    print("batas")
    # myPoints[12:15] = myPoints[12:15] + (myPoints[12:15] * 0.10)
    # myPoints[2:5] = myPoints[2:5] - (myPoints[2:5] * 0.1)
    print(myPoints[12:15])
    imgCheekRight = createBoxLips(img, myPoints[12:15], 2, masked=True, cropped=False)
    imgCheekLeft = createBoxLips(img, myPoints[2:5], 2, masked=True, cropped=False)

    imgColorLipsRight = np.zeros_like(imgCheekRight)
    imgColorLipsRight[:] = b, g, r
    imgColorLipsRight = cv2.bitwise_and(imgCheekRight, imgColorLipsRight)
    imgColorLipsRight = cv2.GaussianBlur(imgColorLipsRight, (15, 15), 10)

    imgColorLipsLeft = np.zeros_like(imgCheekLeft)
    imgColorLipsLeft[:] = b, g, r
    imgColorLipsLeft = cv2.bitwise_and(imgCheekLeft, imgColorLipsLeft)
    imgColorLipsLeft = cv2.GaussianBlur(imgColorLipsLeft, (15, 15), 10)


    imgColorLips = cv2.addWeighted(imgOriginal, 1, imgColorLipsRight, 0.4, 0)
    imgColorLips = cv2.addWeighted(imgColorLips, 1, imgColorLipsLeft, 0.4, 0)
    cv2.imshow('BGR', imgColorLips)


# cv2.imshow('Original', imgOriginal)
cv2.imwrite('hasilEdit.jpg', imgColorLips)
cv2.waitKey(0)
cv2.destroyAllWindows()
