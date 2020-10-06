import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

files1 = [f for f in listdir("G:/mehrnoosh/") if isfile(join("G:/mehrnoosh/", f))]
train_data, labels = [],[] 

for i, file in enumerate(files1):
    image_path = "G:/mehrnoosh/" + files1[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    train_data.append(np.asarray(images, dtype= np.uint8))
    labels.append(i)
    
labels = np.asarray(labels, dtype = np.int32)
model1 = cv2.face.LBPHFaceRecognizer_create()
model1.train(np.asarray(train_data), np.asarray(labels))
print("Model1 trained succesfully")
print(model1)

files2 = [f for f in listdir("G:/mehrnoosh-mask/") if isfile(join("G:/mehrnoosh-mask/", f))]
train_data, labels = [],[] 

for i, file in enumerate(files2):
    image_path = "G:/mehrnoosh-mask/" + files2[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    train_data.append(np.asarray(images, dtype= np.uint8))
    labels.append(i)
    
labels = np.asarray(labels, dtype = np.int32)
model2 = cv2.face.LBPHFaceRecognizer_create()
model2.train(np.asarray(train_data), np.asarray(labels))
print("Model2 trained succesfully")
print(model2)

face_path = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def face_dec(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_path.detectMultiScale(gray, 1.3, 5)
    if faces == ():
        return img, []
    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        r = img[y:y+h, x:x+w]
        r = cv2.resize(r, (200, 200))
        
    return img, r ,(x,y)

capture = cv2.VideoCapture(0)
while True:
    
    try:
        ret, frame = capture.read()
        
        image, face ,(x,y)= face_dec(frame)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model1.predict(face)
        result2 = model2.predict(face)
        
        if result[1] < 500:
            confidence = int(100*(1-(result[1])/300))
            if confidence > 80:
                cv2.putText(image, "Access Denied Mehrnoosh, Please wear your mask", (x-100,y-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow("Face Cropper", image)
            elif confidence < 80:
                cv2.putText(image, "Unknown, Access Denied, Please wear your mask", (x-100,y-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow("Face Cropper", image)
        elif result2[1] < 500:
                 confidence = int(100*(1-(result2[1])/300))
                 if confidence > 80:
                    cv2.putText(image, "Wellcome Mehrnoosh", (x-50,y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Face Cropper", image)
                 else:
                    cv2.putText(image, "Unknown, Wellcome", (x-50,y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Face Cropper", image)

                  
    except:
            print('Error')
            pass
        
    if cv2.waitKey(1) == 13:
        break
    
capture.release()
cv2.destroyAllWindows() 
        
        