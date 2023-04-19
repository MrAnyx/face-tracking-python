# import cv2
# import numpy as np

# faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# cam = cv2.VideoCapture(0)

# ids = int(input("Entrez l'ID de l'utilisateur : "))
# sampleNum = 0

# while(True):
# 	ret, img = cam.read()
# 	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 	faces = faceDetect.detectMultiScale(gray,1.3,5)

# 	for(x,y,w,h) in faces:
# 		sampleNum+=1
# 		# cv2.imwrite("dataSet/User." + str(ids) + "." + str(sampleNum) + ".jpg", gray[y:y+h, x:x+w])
# 		cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
# 	cv2.imshow("Face", img)
# 	if(sampleNum > 19):
# 		break
# cam.release()
# cv2.destroyAllWindows()



import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

ids = 2
sampleNum = 0

while(True):
	ret, img = cam.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in faces:
		sampleNum = sampleNum + 1
		cv2.imwrite("dataSet/User." + str(ids) + "." + str(sampleNum) + ".jpg", gray[y:y+h, x:x+w])
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		cv2.waitKey(60)

	cv2.imshow('img',img)
	cv2.waitKey(1)

	if(sampleNum > 49):
		break

cam.release()
cv2.destroyAllWindows()