import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
finger = [(8, 6), (12, 10), (16, 14), (20, 18)]

while True :
    success, img = cap.read()
    img = cv2.flip(img, 1)
    #print(success) #shows that if we are getting img or not
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB) #contains the hand in RGB
    multiLandMarks = results.multi_hand_landmarks #returns a list of coordinates for each finger
    #print(multiLandMarks)

    if multiLandMarks : #only if hand is present
        handPoints = []
        for handLms in multiLandMarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) #shows landmark for each hand coordinates

            for idx, lm in enumerate(handLms.landmark) : #calculates coordinates from 0 to 20 for each hand
                #print(idx,lm)
                h,w,c = img.shape
                cx,cy = int(lm.x * w), int(lm.y * h)
                #print(cx,cy) #prints number of pixel from top left corner
                handPoints.append((cx, cy))
        
        count = 0
        for points in finger :
            if handPoints[points[0]][1] < handPoints[points[1]][1] : #y coordinate
                count += 1
        if handPoints[1][0] < handPoints[0][0] : #checks the postioning of hand whether right or left wrt to hand facing screen, this case is for right hand
            if handPoints[4][0] < handPoints[2][0] : # for thumb x axis, works when right hand facing screen
                count += 1
        else :#for left hand
            if handPoints[4][0] > handPoints[2][0] : # for thumb x axis, works when left hand facing screen
                count += 1

        #print(count)
        cv2.putText(img, str(count), (150,150), cv2.FONT_HERSHEY_PLAIN, 12, (255,0,0), 12)

    cv2.imshow('Finger Counter', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
