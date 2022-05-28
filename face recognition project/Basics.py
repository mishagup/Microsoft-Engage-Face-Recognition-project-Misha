import cv2
import face_recognition

imgmisha = face_recognition.load_image_file('images/misha.jpeg')
imgmisha = cv2.cvtColor(imgmisha, cv2.COLOR_BGR2RGB)
imgmisha1 = face_recognition.load_image_file('images/misha1.jpeg')
imgmisha1 = cv2.cvtColor(imgmisha1, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgmisha)[0]
encodemisha = face_recognition.face_encodings(imgmisha)[0]
cv2.rectangle(imgmisha,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocmisha1 = face_recognition.face_locations(imgmisha1)[0]
encodemisha1 = face_recognition.face_encodings(imgmisha1)[0]
cv2.rectangle(imgmisha1,(faceLocmisha1[3],faceLocmisha1[0]),(faceLocmisha1[1],faceLocmisha1[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodemisha], encodemisha1)
faceDis = face_recognition.face_distance([encodemisha], encodemisha1)
print(results, faceDis)
cv2.putText(imgmisha1, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('misha',imgmisha)
cv2.imshow('misha1',imgmisha1)
cv2.waitKey(0)