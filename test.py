import cv2
#
cap=cv2.VideoCapture(8)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 30)
# cap2=cv2.VideoCapture(4)
key=0
while not (key==13):
    ret, frame = cap.read()
    if ret:
        cv2.imshow("p", frame)
        key=cv2.waitKey(30)
for i in range(100):
    ret,frame=cap.read()
    if ret:
        cv2.imshow("p",frame)
        cv2.waitKey(30)
        cv2.imwrite("r/"+str(i)+".jpg",frame)
