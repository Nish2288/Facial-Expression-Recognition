import cv2
import numpy as np 
import os





def main():
    #capture_digit("Training_Data/" + str(input_digit))
    x, y, w, h =300, 50, 300, 300
    max_pics = 1000
    image_no=0
    cap = cv2.VideoCapture(0)

    while True :
        ret,frame = cap.read()
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        lower = np.array([2, 50, 60], dtype = "uint8")
        upper = np.array([25, 150, 255], dtype = "uint8")
        
        mask1 = cv2.inRange(hsv, lower, upper)
        frame2 = cv2.bitwise_and(frame, frame, mask=mask1)
        gray = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 

        
        #kernel = np.ones((5,5),np.uint8)
        
        #dilation = cv2.dilate(gray,kernel,iterations = 2)
        #opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel)

        ret, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        thresh = thresh[y:y+h, x:x+w]
        contours = cv2.findContours(thresh.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
        if len(contours) > 0 :
            contour = max(contours, key = cv2.contourArea)
            if cv2.contourArea(contour)>5000 :
                x1, y1, w1, h1 =cv2.boundingRect(contour)
                image=thresh[y1:y1+h1, x1:x1+w1]
                if w1 > h1:
                    image = cv2.copyMakeBorder(image, int((w1 - h1) / 2), int((w1 - h1) / 2), 0, 0,
                                                  cv2.BORDER_CONSTANT, (0, 0, 0))
                elif h1 > w1:
                    image = cv2.copyMakeBorder(image, 0, 0, int((h1 - w1) / 2), int((h1 - w1) / 2),
                                                  cv2.BORDER_CONSTANT, (0, 0, 0))
                image=cv2.resize(image,(100,100))
                image_no+=1
                #cv2.imwrite("Training_Data/" + str(input_digit)+"/"+str(image_no)+ ".jpg",image)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, str(image_no), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
        cv2.imshow("Record Number", frame)
        cv2.imshow("Computer Vision", thresh)
        
        
        if image_no==1000:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


#input_digit=input('Enter digit to capture :')
main()

