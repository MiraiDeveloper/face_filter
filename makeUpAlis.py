import cv2
import numpy as np
import dlib
import streamlit as st


cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

r = 255
g = 192
b = 203

def createBoxLips(img, points, scale=5, masked=False, cropped=True):
    if masked:
        mask = np.zeros_like(img)
        #block
        # mask = cv2.fillPoly(mask, [points], (255, 255, 255))
        #line
        mask = cv2.polylines(mask, [points], False, (255, 255, 255), thickness=5)
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
st.write('Gambar Sebelum')
st.image('woman.jpg')

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
        myPoints.append([x, y+0])
        # tampilan shape predictor
        # print(myPoints[n])
        # cv2.circle(imgOriginal, (x, y), 2, (50, 50, 255), cv2.FILLED)
        # cv2.putText(imgOriginal, str(n), (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1)

    myPoints = np.array(myPoints)
    # print(myPoints[48:61])
    imgEyeBrowLeft = createBoxLips(img, myPoints[17:22], 2, masked=True, cropped=False)
    imgEyeBrowRight = createBoxLips(img, myPoints[22:27], 2, masked=True, cropped=False)
    # cv2.imshow('test', imgEyeBrowRight)

    imgColorEyeBrowLeft = np.zeros_like(imgEyeBrowLeft)
    imgColorEyeBrowLeft[:] = b, g, r
    imgColorEyeBrowLeft = cv2.bitwise_and(imgEyeBrowLeft, imgColorEyeBrowLeft)
    imgColorEyeBrowLeft = cv2.GaussianBlur(imgColorEyeBrowLeft, (7, 7), 10)

    imgColorEyeBrowRight = np.zeros_like(imgEyeBrowRight)
    imgColorEyeBrowRight[:] = b, g, r
    imgColorEyeBrowRight = cv2.bitwise_and(imgEyeBrowRight, imgColorEyeBrowRight)
    imgColorEyeBrowRight = cv2.GaussianBlur(imgColorEyeBrowRight, (7, 7), 10)

    imgColorEyeBrows = cv2.addWeighted(imgOriginal, 1, imgColorEyeBrowLeft, 0.4, 0)
    imgColorEyeBrows = cv2.addWeighted(imgColorEyeBrows, 1, imgColorEyeBrowRight, 0.4, 0)
    cv2.imshow('BGR', imgColorEyeBrows)

# cv2.imshow('Original', imgOriginal)
cv2.imwrite('../../hasilEdit.jpg', imgColorEyeBrows)
st.write('Gambar Sesudah')
st.image('hasilEdit.jpg')
cv2.waitKey(0)
cv2.destroyAllWindows()
