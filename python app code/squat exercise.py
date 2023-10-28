import cv2
import mediapipe as mp
import numpy as np
import math
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)

def calculate_angle1(a,b,c):
    np.array(a) 
    np.array(b) 
    np.array(c) 
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle1 = np.abs(radians*180.0/np.pi)
    if angle1 >180.0:
        angle1 = 360-angle1
    return math.floor(angle1)



def calculate_angle2(d,e,f):
    np.array(d)
    np.array(e)
    np.array(f)
    radians = np.arctan2(f[1]-e[1], f[0]-e[0]) - np.arctan2(d[1]-e[1], d[0]-e[0])
    angle2 = np.abs(radians*180.0/np.pi)
    if angle2>180.0:
        angle2=360-angle2
    return math.floor(angle2)

def calculate_angle3(g,a,b):
    np.array(g)
    np.array(a)
    np.array(b)
    radians = np.arctan2(b[1]-a[1], b[0]-a[0]) - np.arctan2(g[1]-a[1], g[0]-a[0])
    angle3 = np.abs(radians*180.0/np.pi)
    if angle3>180.0:
        angle3=360-angle3
    return math.floor(angle3)

def calculate_angle4(h,d,e):
    np.array(h)
    np.array(d)
    np.array(e)
    radians = np.arctan2(e[1]-d[1], e[0]-d[0]) - np.arctan2(h[1]-d[1], h[0]-d[0])
    angle4 = np.abs(radians*180.0/np.pi)
    if angle4>180.0:
        angle4=360-angle4
    return math.floor(angle4)




# Curl counter variables
counter = 0 
stage = None


## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            hipL = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            kneeL = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankleL = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            hipR = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            kneeR = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankleR = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            shoulderL = [landmarks[mp_pose.poseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            shoulderR = [landmarks[mp_pose.poseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

            
 
    
            
            
            # Calculate angle
            angle1 = calculate_angle1(hipL, kneeL, ankleL)
            angle2 = calculate_angle2(hipR, kneeR, ankleR)
            angle3 = calculate_angle3(shoulderL, hipL, kneeL)
            angle4 = calculate_angle4(shoulderR, hipR, kneeR)
            print('left angle knee',angle1)
            print('right angle knee',angle2)
            print('left angle hip',angle3)
            print('right angle hip',angle4)


            # Visualize angle
            cv2.putText(image, str(angle1), 
                           tuple(np.multiply(kneeL, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, str(angle2), 
                           tuple(np.multiply(kneeR, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, str(angle3), 
                           tuple(np.multiply(hipL, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, str(angle4), 
                           tuple(np.multiply(hipR, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )

            # Curl counter logic
            if angle1+angle2> 220:
                stage = "down"
            if angle1+angle2 < 150 and stage =='down':
                stage="up"
                counter +=1
                print(counter)

            if angle3+angle4> 300 :
                stage = "down"
            if angle1+angle2 < 120 and stage =='down':
                stage="up"
                counter +=1
                print(counter)

            
            




        except:
            pass


        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
        # Rep data
        cv2.putText(image, 'REPS', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # Stage data
        cv2.putText(image, 'STAGE', (65,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, 
                    (60,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
       
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

