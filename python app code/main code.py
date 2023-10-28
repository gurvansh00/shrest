import cv2
import mediapipe as mp
import numpy as np
import math
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(r"C:\Users\Dell\Downloads\Lifestyle stock footage of Man running against silhouetted mountain.mp4")

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


def calculate_angle5(i,g,a):
    np.array(i)
    np.array(g)
    np.array(a)
    radians = np.arctan2(a[1]-g[1], a[0]-g[0]) - np.arctan2(i[1]-g[1], i[0]-g[0])
    angle5 = np.abs(radians*180.0/np.pi)
    if angle5>180.0:
        angle5=360-angle5
    return math.floor(angle5)


def calculate_angle6(j,h,d):
    np.array(j)
    np.array(h)
    np.array(d)
    radians = np.arctan2(d[1]-h[1], d[0]-h[0]) - np.arctan2(j[1]-h[1], j[0]-h[0])
    angle6 = np.abs(radians*180.0/np.pi)
    if angle6>180.0:
        angle6=360-angle6
    return math.floor(angle6)


def calculate_angle7(k,i,g):
    np.array(k)
    np.array(i)
    np.array(g)
    radians = np.arctan2(g[1]-i[1], g[0]-i[1]) - np.arctan2(k[1]-i[1], k[0]-i[0])
    angle7 = np.abs(radians*180.0/np.pi)
    if angle7>180.0:
        angle7=360-angle7
    return math.floor(angle7)


def calculate_angle8(l,j,h):
    np.array(l)
    np.array(j)
    np.array(h)
    radians = np.arctan2(h[1]-j[1], h[0]-j[1]) - np.arctan2(l[1]-j[1], l[0]-j[0])
    angle8 = np.abs(radians*180.0/np.pi)
    if angle8>180.0:
        angle8=360-angle8
    return math.floor(angle8)



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
            shoulderL = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbowL = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wristL = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            shoulderR = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbowR = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wristR = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            hipL = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            hipR = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            kneeL = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            kneeR = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankleL = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            ankleR = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
 
    
            
            
            # Calculate angle
            angle1 = calculate_angle1(shoulderL, elbowL, wristL)
            angle2 = calculate_angle2(shoulderR, elbowR, wristR)
            angle3 = calculate_angle3(hipL, shoulderL, elbowL)
            angle4 = calculate_angle4(hipR, shoulderR, elbowR)
            angle5 = calculate_angle5(kneeL, hipL, shoulderL)
            angle6 = calculate_angle6(kneeR, hipR, shoulderR)
            angle7 = calculate_angle7(ankleL, kneeL, hipL)
            angle8 = calculate_angle8(ankleR, kneeR, hipR)
            print('left angle elbow',angle1)
            print('right angle elbow',angle2)
            print('left angle shoulder',angle3)
            print('right angle shoulder',angle4)
            print('left angle hip',angle5)
            print('right angle hip',angle6)
            print('left angle knee',angle7)
            print('right angle knee',angle8)


            # Visualize angle
            cv2.putText(image, str(angle1), 
                           tuple(np.multiply(elbowL, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, str(angle2), 
                           tuple(np.multiply(elbowR, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, str(angle3), 
                           tuple(np.multiply(shoulderL, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, str(angle4), 
                           tuple(np.multiply(shoulderR, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, str(angle5), 
                           tuple(np.multiply(hipL, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, str(angle6), 
                           tuple(np.multiply(hipR, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, str(angle7), 
                           tuple(np.multiply(kneeL, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, str(angle8), 
                           tuple(np.multiply(kneeR, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )






        except:
            pass


       
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


