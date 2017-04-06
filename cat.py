# -*- coding=utf-8 -*-
import cv2
# 加载猫脸检测器
catPath = "haarcascade_frontalcatface.xml"
faceCascade = cv2.CascadeClassifier(catPath)
# 读取图片并灰度化
img = cv2.imread("cat1.jpg")  
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 猫脸检测
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor= 1.02,
    minNeighbors=3,
    minSize=(150, 150),
    flags=cv2.CASCADE_SCALE_IMAGE
)
# 框出猫脸并加上文字说明
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.putText(img,'Cat',(x,y-7), 3, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
# 显示图片并保存
cv2.imshow('Cat?', img)
cv2.imwrite("cat33.jpg",img)
c = cv2.waitKey(0)