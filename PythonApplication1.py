import cv2
face_clissifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def F1(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_clissifier.detectMultiScale(gray,1.1,5)
    if(faces == ()):
        return ()
    (x,y,w,h) = faces[0]
    cropped_image = gray[y:y + h, x:x + w]
    return cropped_image

cam = cv2.VideoCapture(0)
counter = 1
while True:
    status, frame = cam.read()
    if(status == True):
        face = F1(frame)
        if(face != ()):
            face = cv2.resize(face,(200,200))
            cv2.imwrite(f'G:/mehrnoosh/user{counter}.jpg',face)
            cv2.putText(face,str(counter),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            cv2.imshow('samples',face)
            counter+=1
        
    if cv2.waitKey(1) == 27 or counter == 201:
        break

cam.release()
cam = cv2.VideoCapture(0)
counter = 201
while True:
    status, frame = cam.read()
    if(status == True):
        face = F1(frame)
        if(face != ()):
            face = cv2.resize(face,(200,200))
            cv2.imwrite(f'G:/mehrnoosh-mask/user{counter}.jpg',face)
            cv2.putText(face,str(counter)+'please wear your mask',(1,50),cv2.FONT_HERSHEY_COMPLEX,0.45,(0,0,0),2)
            cv2.imshow('samples',face)
            counter+=1
      
    if cv2.waitKey(1) == 27 or counter == 401:
        break
cam.release()