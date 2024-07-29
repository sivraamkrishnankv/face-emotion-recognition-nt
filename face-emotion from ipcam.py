##from facial_emotion_recognition import EmotionRecognition
##import cv2
##import urllib.request
##import numpy as np
##import imutils
##
##er=EmotionRecognition(device='cpu')
##
##url = 'http://192.168.0.109:8080/shot.jpg'
##
##while True:
##     imgPath = urllib.request.urlopen(url)
##     imgNp = np.array(bytearray (imgPath.read()), dtype = np.uint8)
##     frame = cv2.imdecode(imgNp,-1)
##     frame = er.recognise_emotion(frame, return_type='BGR')
##     frame = imutils.resize(frame, width=850)
##     cv2.imshow("Frame",frame)
##     
##     key = cv2.waitKey(1)
##     if key == 27:
##         break
##cam.release()
##cv2.destroyAllWindows()


                        #chat gpt code

from facial_emotion_recognition import EmotionRecognition
import cv2
import urllib.request
import numpy as np
import imutils

er = EmotionRecognition(device='cpu')

url = 'http://192.168.0.100:8080/shot.jpg'

while True:
    imgPath = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgPath.read()), dtype=np.uint8)
    frame = cv2.imdecode(imgNp, -1)
    frame = er.recognise_emotion(frame, return_type='BGR')

    # Enlarge the text and change its color to blue
    emotions = er.emotions
    for idx, (emotion, _) in enumerate(emotions.items()):
        text = f"{emotion}: {_}"
        cv2.putText(frame, text, (10, 30 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    frame = imutils.resize(frame, width=850)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()

