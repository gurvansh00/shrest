from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import time
import io
from PIL import Image
import base64,cv2
import numpy as np
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



bicep_curl_count = 0
accurate_curl_count = 0
total_curl_count = 0


app = Flask(__name__)
socketio = SocketIO(app,cors_allowed_origins='*' )



@app.route('/', methods=['POST', 'GET'])

def index():
    return render_template('index.html')


def readb64(base64_string):
    idx = base64_string.find('base64,')
    base64_string  = base64_string[idx+7:]

    sbuf = io.BytesIO()

    sbuf.write(base64.b64decode(base64_string, ' /'))
    pimg = Image.open(sbuf)


    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

def moving_average(x):
    return np.mean(x)


@socketio.on('catch-frame')
def catch_frame(data):

    emit('response_back', data)  


global fps,prev_recv_time,cnt,fps_array
fps=30
prev_recv_time = 0
cnt=0
fps_array=[0]

@socketio.on('image')
def image(data_image):
    global fps,cnt, prev_recv_time,fps_array
    recv_time = time.time()
    text  =  'FPS: '+str(fps)
    frame = (readb64(data_image))

    frame = changeLipstick(frame,[255,0,0])
    frame = ps.putBText(frame,text,text_offset_x=20,text_offset_y=30,vspace=20,hspace=10, font_scale=1.0,background_RGB=(10,20,222),text_RGB=(255,255,255))
    imgencode = cv2.imencode('.jpeg', frame,[cv2.IMWRITE_JPEG_QUALITY,40])[1]

    # base64 encode
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpeg;base64,'
    stringData = b64_src + stringData

    # emit the frame back
    emit('response_back', stringData)
    
    fps = 1/(recv_time - prev_recv_time)
    fps_array.append(fps)
    fps = round(moving_average(np.array(fps_array)),1)
    prev_recv_time = recv_time
    #print(fps_array)
    cnt+=1
    if cnt==30:
        fps_array=[fps]
        cnt=0
    



def getMaskOfLips(img,points):
    """ This function will input the lips points and the image
        It will return the mask of lips region containing white pixels
    """
    mask = np.zeros_like(img)
    mask = cv2.fillPoly(mask,[points],(255,255,255))
    return mask 


def changeLipstick(img,value):
    """ This funciton will take img image and lipstick color RGB
        Out the image with a changed lip color of the image
    """ 

    img = cv2.resize(img,(0,0),None,1,1)
    imgOriginal = img.copy()
    imgColorLips=imgOriginal
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(imgGray)

    for face in faces:
        x1,y1 = face.left(),face.top()
        x2,y2 = face.right(),face.bottom()
       
        facial_landmarks = predictor(imgGray,face)
        points =[]
        for i in range(68):
            x = facial_landmarks.part(i).x
            y = facial_landmarks.part(i).y
            points.append([x,y])


        points = np.array(points)
        imgLips = getMaskOfLips(img,points[48:61])
        
        imgColorLips = np.zeros_like(imgLips)
        
        imgColorLips[:] =value[2],value[1],value[0]
        imgColorLips = cv2.bitwise_and(imgLips,imgColorLips)
        
        value = 1
        value=value//10
        if value%2==0:
            value+=1
        kernel_size = (6+value,6+value) # +1 is to avoid 0
        
        weight = 1
        weight = 0.4 + (weight)/400
        imgColorLips = cv2.GaussianBlur(imgColorLips,kernel_size,10)
        imgColorLips = cv2.addWeighted(imgOriginal,1,imgColorLips,weight,0)
        
        
        
    return imgColorLips


if __name__ == '__main__':
    socketio.run(app,port=0 ,debug=True)
   


