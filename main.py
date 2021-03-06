import cv2
import numpy as np
import face_recognition

#convert to rgb as library understands only rgb and the raw input is of bgr


imgElon=face_recognition.load_image_file('imageTest/em.jpg')
imgElon=cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)

imgTest=face_recognition.load_image_file('imageTest/images.jpg')
imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)


#finding the faces and findinng their encodings too

faceLoc=face_recognition.face_locations(imgElon)[0]
encodeElon= face_recognition.face_encodings(imgElon)[0]
#on printing the face location returns 4 values
#in my case (168, 425, 297, 296) this was the output
#these values are top right bottom and left
#based on these i can give x and y axis

# print(faceLoc)

cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)


#Doing for test
faceLocTest=face_recognition.face_locations(imgTest)[0]
encodeTest= face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)


#we will use the linear SVM to match
#list of known faces is passed
results=face_recognition.compare_faces([encodeElon],encodeTest)
faceDis=face_recognition.face_distance([encodeElon],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow("Elon",imgElon)
cv2.imshow("Test",imgTest)
cv2.waitKey(0)


