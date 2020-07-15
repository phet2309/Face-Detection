import cv2
import numpy as np
# print(cv2.__version__)

video_capture=cv2.VideoCapture(0)
fourcc=cv2.VideoWriter_fourcc(*'XVID')
# out=cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')


while True:
    _, img=video_capture.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    # out.write(img)
    # cv2.imshow("Hello_1",img)
    # cv2.imshow("Hello",gray)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h), (255,0,0), 2)
        roi_gray=gray[y:y+h, x:x+w]
        roi_color=img[y:y+h, x:x+w]
        eyes=eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)



    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
# out.release()
cv2.destroyAllWindows()