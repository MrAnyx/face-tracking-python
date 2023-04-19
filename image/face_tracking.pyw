import cv2

# faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")

# img=cv2.imread('image.jpg')

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# faces = faceCascade.detectMultiScale(gray,scaleFactor= 1.3,minNeighbors=3,minSize=(55, 55))
# eyes = eyeCascade.detectMultiScale(gray,scaleFactor= 1.3,minNeighbors=3,minSize=(10, 10))


# for (x, y, w, h) in faces:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

# for (x, y, w, h) in eyes:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

              
               
# cv2.imshow('Face_Detector', img)
# cv2.waitKey(0)


imagePath = "test.jpg"
cascPath = "D:/Programme_perso/Face_detection_Lib/haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascPath)

image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30,30),
)

print("Found {0} faces!".format(len(faces)))


for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
cv2.destroyAllWindows()