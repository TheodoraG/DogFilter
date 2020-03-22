
import numpy as np
import cv2

class FaceFilter:
    def __init__(self):
        self.__inputImage = cv2.imread("human.jpg")

    def __ConstructBlob(self, net):
        blob = cv2.dnn.blobFromImage(cv2.resize(self.__inputImage, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)

    def __ApplyDogFilter(self, dog, img, x, y, w, h):
        faceWidth = w
        faceHight =  h
        dog = cv2.resize(dog, (int(faceWidth*1.75), int(faceHight*1.75)))
        for i in range(int(faceHight*1.75)):
            for j in range(int(faceWidth*1.5)):
                for k in range(3):
                    if dog[i][j][k] < 235:
                       img[y+i-int(0.2*h)-1][x+j-int(0.45*w)][k] = dog[i][j][k]
        return img

    def __PerformFaceDetection(self, net, h, w, dog):
        detections = net.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0,0,i,2]
            if confidence > 0.5:
                box = detections[0,0,i,3:7] * np.array([w,h,w,h])
                (startX, startY, endX, endY) = box.astype("int")

                y = startY - 10 if startY - 10 > 10 else startY + 10
                #cv2.rectangle(self.__inputImage, (startX, startY), (endX, endY), (0, 0, 255), 2)
                self.__inputImage = self.__ApplyDogFilter(dog, self.__inputImage, startX, startY, endX-startX, endY-startY)

    def ShowResults(self):
        net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt","res10_300x300_ssd_iter_140000.caffemodel")
        dog = cv2.imread("dog_filter.png")
        h = self.__inputImage.shape[0]
        w = self.__inputImage.shape[1]
        self.__ConstructBlob(net)
        self.__PerformFaceDetection(net,h,w,dog)
        #self.__inputImage = cv2.resize(self.__inputImage,(400,500))
        #cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        cv2.imshow("Image", self.__inputImage)
        cv2.waitKey(0)

objectClass = FaceFilter()
objectClass.ShowResults()