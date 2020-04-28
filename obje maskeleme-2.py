import cv2
import numpy as np
import fonksiyon


kernel = np.ones((5,5),np.uint8)
cv2.namedWindow("Staked Images",cv2.WINDOW_NORMAL)
frameWidth=480
frameHeight =360
cap=cv2.VideoCapture(0)


cap.set(3,frameWidth)
cap.set(4,frameWidth)

def empty(a):
    pass

cv2.namedWindow("HSV")
cv2.resizeWindow("HSV",640,240)
#cv2.namedWindow("HSV",cv2.WINDOW_NORMAL)
cv2.createTrackbar("HUE Min","HSV",0,179,empty)
cv2.createTrackbar("HUE Max","HSV",179,179,empty)
cv2.createTrackbar("SAT Min","HSV",0,255,empty)
cv2.createTrackbar("SAT Max","HSV",255,255,empty)
cv2.createTrackbar("Value Min","HSV",0,255,empty)
cv2.createTrackbar("Value Max","HSV",0,255,empty)



while True:
    _,img= cap.read()
    imgHsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("HUE Min","HSV")
    h_max = cv2.getTrackbarPos("HUE Max","HSV")
    s_min = cv2.getTrackbarPos("SAT Min","HSV")
    s_max = cv2.getTrackbarPos("SAT Max","HSV")
    v_min = cv2.getTrackbarPos("Value Min","HSV")
    v_max = cv2.getTrackbarPos("Value Max","HSV")
    #print(h_min)


    lower = np.array([ h_min ,s_min , v_min])
    upper = np.array([ h_max, s_max , v_max])
    mask = cv2.inRange(imgHsv,lower,upper)
    result = cv2.bitwise_and(img,img,mask=mask)
    mask=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    imgGen = cv2.dilate(result,kernel,4)
    imgDar = cv2.erode(result,kernel,4)
    imgBugu = cv2.GaussianBlur(result,(7,7),5)

    
    cv2.putText(img,"orijinal",(430,460),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    cv2.putText(imgHsv,"HSV image",(430,460),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    cv2.putText(mask,"Mask",(430,460),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    cv2.putText(result,"Sonuc",(430,460),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    cv2.putText(imgGen,"Genisletme",(430,460),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    cv2.putText(imgDar,"Daraltma",(430,460),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    cv2.putText(imgBugu,"Bugulama",(430,460),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    #hstack =  np.hstack([img,mask,result])
    
    #cv2.imshow("original",img)
    #cv2.imshow("HSV image",imgHsv)
    #cv2.imshow("mask",mask)
    #cv2.imshow("result",result)
    #
    #cv2.imshow("genisletilmis",imgGen )
    #cv2.imshow("daraltma",imgDar)
    #cv2.imshow("genel",hstack)
    #imgBlank = np.zeros((200,200),np.uint8)
    
    StackedImages = fonksiyon.stackImages(0.6,([img,result,mask],[imgBugu,imgGen,imgDar]))
    cv2.imshow("Staked Images", StackedImages)

    if cv2.waitKey(1) & 0xff == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
