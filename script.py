import cv2
import glob

face_cascade= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

images = glob.glob("*.jpg")
for image in images:
    if ("detected_" in image):
        continue
    img= cv2.imread(image)
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces= face_cascade.detectMultiScale(gray_img,
    scaleFactor=1.03,
    minNeighbors=5
    )

    for x, y, w, h in faces:
        img= cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3 )

    cv2.imwrite("detected_"+image, img)
