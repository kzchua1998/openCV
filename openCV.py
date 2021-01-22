#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# # Reading photo

# In[5]:


img = cv2.imread('image/hell.jpg')
cv2.imshow('photo', img)
cv2.waitKey(2000)
cv2.destroyWindow('photo')


# # Reading video

# In[5]:


i = 0
vid = cv2.VideoCapture('nazi.mp4')
while True:
    success, img_slice = vid.read()
    cv2.imshow('video', img_slice)
    i += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyWindow('video')
print(i)


# # Webcam

# In[32]:


vid = cv2.VideoCapture(0)
vid.set(3, 1000)
vid.set(4, 1000)
vid.set(10, 100)
while True:
    success, img_slice = vid.read()
    cv2.imshow('video', img_slice)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyWindow('video')


# # Functions

# ### Grayscale 

# In[11]:


img = cv2.imread('image/hell.jpg')
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray_photo', imgGray)
cv2.waitKey(0)


# ### Blur 

# In[3]:


img = cv2.imread('image/hell.jpg')
imgBlur = cv2.GaussianBlur(imgGray, (11,11), 0)
cv2.imshow('gray_blur_photo', imgBlur)
cv2.waitKey(0)


# ### Edge Detector 

# In[13]:


img = cv2.imread('image/hell.jpg')
imgCanny = cv2.Canny(img, 100, 100)
cv2.imshow('canny_photo', imgCanny)
cv2.waitKey(0)


# ### Morphological Transformations (from Canny)

# In[15]:


img = cv2.imread('image/hell.jpg')
imgCanny = cv2.Canny(img, 100, 100)
kernel = np.ones((2,2), np.uint8)
# dilation
imgDilation = cv2.dilate(imgCanny, kernel, iterations = 1)
cv2.imshow('dilation_photo', imgDilation)
# erosion
imgErosion = cv2.erode(imgDilation, kernel, iterations = 1)
cv2.imshow('erosion_photo', imgErosion)
# opening (erode then dilate)
imgOpening = cv2.morphologyEx(imgCanny, cv2.MORPH_OPEN, kernel)
cv2.imshow('opening_photo', imgOpening)
# closing (dilste then erode)
imgClosing = cv2.morphologyEx(imgCanny, cv2.MORPH_CLOSE, kernel)
cv2.imshow('closing_photo', imgClosing)
cv2.waitKey(0)


# # Resizing and Cropping

# In[18]:


img = cv2.imread('image/hell.jpg')
print(img.shape)
imgResize = cv2.resize(img, (500, 400))
cv2.imshow('resize_photo', imgResize)
cv2.waitKey(0)
cv2.destroyWindow('photo')


# In[20]:


imgCropped = img[0:200, 300:500]
cv2.imshow('resize_photo', imgCropped)
cv2.waitKey(0)
cv2.destroyWindow('photo')


# # Shapes and Texts

# In[35]:


img = np.zeros((512,512,3), np.uint8)
#img[:] = 0, 0, 255
cv2.line(img, (0, 0), (300, 300), (100, 34, 67), 3)
cv2.rectangle(img, (0, 0), (250, 250), (100, 94, 167), cv2.FILLED)
cv2.circle(img, (0, 0), 100, (1, 67, 45), cv2.FILLED)
cv2.putText(img, 'opencv', (300, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (100, 100, 100), 1)
cv2.imshow('color_photo', img)
cv2.waitKey(0)


# # Warp Perspective

# In[16]:


img = cv2.imread('image/business_card.jfif')
img_cropped = img[:, 100:1100]
width, height = 500, 500
# use paint to locate the points
pts1 = np.float32([[140, 190],[540, 60],[430, 413],[837, 280]])
pts2 = np.float32([[0, 0],[width, 0],[0, height],[width, height]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
imgOutput = cv2.warpPerspective(img_cropped, matrix, (width, height))
cv2.imshow('business_card', img_cropped)
cv2.imshow('business_card1', imgOutput)
cv2.waitKey(0)


# # Joining Images

# In[2]:


img = cv2.imread('image/hell.jpg')
hor = np.hstack((img, img))
ver = np.vstack((img, img))
cv2.imshow('horizontal_photo', hor)
cv2.imshow('vertical_photo', ver)
cv2.waitKey(0)


# In[2]:


# to be able to stack arrays of images with different channel number, a function is needed
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


# # Color Detection

# In[15]:


img = cv2.imread('bmw330.jpeg')
imgResize = cv2.resize(img, (500, 400))
imgHSV = cv2.cvtColor(imgResize, cv2.COLOR_BGR2HSV)
cv2.imshow('bmw_photo', imgHSV)
cv2.waitKey(0)


# ### Track Bars

# In[3]:


img = cv2.imread('bmw330.jpeg')
imgResize = cv2.resize(img, (500, 400))

cv2.namedWindow('TrackBars')
cv2.resizeWindow('TrackBars', 500, 400)
imgHSV = cv2.cvtColor(imgResize,cv2.COLOR_BGR2HSV)

# to get pass
def empty(a):
    pass

cv2.createTrackbar("Hue Min","TrackBars",0,179,empty)
cv2.createTrackbar("Hue Max","TrackBars",179,179,empty)
cv2.createTrackbar("Sat Min","TrackBars",0,255,empty)
cv2.createTrackbar("Sat Max","TrackBars",255,255,empty)
cv2.createTrackbar("Val Min","TrackBars",0,255,empty)
cv2.createTrackbar("Val Max","TrackBars",255,255,empty)

while True:
    h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHSV,lower,upper)
    imgResult = cv2.bitwise_and(imgResize,imgResize,mask=mask)

    imgStack = stackImages(0.6,([imgResize,imgHSV],[mask,imgResult]))
    cv2.imshow("Stacked Images", imgStack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyWindow('Stacked Images')
cv2.destroyWindow('TrackBars')


# # Face Detection

# In[14]:


faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("Resources/haarcascade_eye.xml")
array = [(faceCascade, 'face'), (eyeCascade, 'eye')]
vid = cv2.VideoCapture(0)
vid.set(3, 1000)
vid.set(4, 1000)
vid.set(10, 100)
while True:
    success, img_slice = vid.read()
    imgGray = cv2.cvtColor(img_slice,cv2.COLOR_BGR2GRAY)
    for cascade, feature_name in array:
        feature = cascade.detectMultiScale(imgGray,1.5,5)
        for (x,y,w,h) in feature:
            cv2.rectangle(img_slice,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(img_slice,feature_name,
                       (x+(w//2),y+(h//2)),cv2.FONT_HERSHEY_COMPLEX,0.7,
                       (0,255,0),1)
    cv2.imshow('video', img_slice)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyWindow('video')


# In[ ]:




