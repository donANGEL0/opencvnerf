import cv2
import numpy as np
import pickle
import serial
import time
import sys

face_cascade = cv2.CascadeClassifier('C:/Python/cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

#arduino = serial.Serial('COM3', 9600, timeout=2)

labels = {"person_name": 1} 
with open("face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while True:
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, minNeighbors=5)
	for (x, y, w, h) in faces:
		#print(x,y,) 
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+w]
		xx = int(x+(x+h))/2
		yy = int(y+(y+w))/2
		#print (xx)
		#print (yy)
		center = (xx,yy)
		print(center)

		id_, conf = recognizer.predict(roi_gray)
		if conf>=45 and conf<=85:
			#print(id_)
			#print(labels[id_])
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color = (255, 255, 255)
			stroke = 2
			cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

		#print("Center of Rectangle is :", center)
		data = "<X{0:}||Y{1:}>".format(xx, yy)
		#print(data)
		#print(type(data))

		#bindata = ''.join(format(ord(i), 'b') for i in data)
		#print(bindata)
		#print(type(bindata))

		bdata = data.encode('ascii')
		#print(bdata)
		#print(type(bdata))
		#arduino.write(bdata)


		color = (255, 0, 0)
		stroke = 2
		width = x + w
		height = y + h
		cv2.rectangle(frame, (x, y), (width, height), color, stroke)



	cv2.imshow('frame', frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()     