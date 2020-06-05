from cv2 import cv2
import numpy as np
import pickle

#Camera Resolution
frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0.75
font = cv2.FONT_HERSHEY_SIMPLEX

#Setup The Video Camera
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

#Import the model 
pickle_in = open('model_trained.p', 'rb')
model = pickle.load(pickle_in)

def grayScale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayScale(img)
    img = equalize(img)
    img = img/255
    return img

def getClassName(classNo):
    if classNo == 0:
        return 'Speed Limit 20 km/h'
    elif classNo == 1:
        return 'Speed Limit 30 km/h'
    elif classNo == 2:
        return 'Speed Limit 50 km/h'
    elif classNo == 3:
        return 'Speed Limit 60 km/h'
    elif classNo == 4:
        return 'Speed Limit 70 km/h'
    elif classNo == 5:
        return 'Speed Limit 80 km/h'
    elif classNo == 6:
        return 'End OfSpeed Limit 80 km/h'
    elif classNo == 7:
        return 'Speed Limit 100 km/h'
    elif classNo == 8:
        return 'Speed Limit 120 km/h'
    elif classNo == 9:
        return 'No Passing'
    elif classNo == 10:
        return 'No Passing For Vehicles Over 3.5 Metric Tons'
    elif classNo == 11:
        return 'Right-Of-Way At The Next Intersection'
    elif classNo == 12:
        return 'Priority Road
    elif classNo == 13:
        return 'Yeild'
    elif classNo == 14:
        return 'Stop'
    elif classNo == 15:
        return 'No Vehicles'
    elif classNo == 16:
        return 'Vehicles Over 3.5 Metric Tons Prohibited'
    elif classNo == 17:
        return 'No Entry'
    elif classNo == 18:
        return 'General Caution'
    elif classNo == 19:
        return 'Dangerous Curve To The Left'
    elif classNo == 20:
        return 'Dangerous Curve To The Right'
    elif classNo == 21:
        return 'Double Curve'
    elif classNo == 22:
        return 'Bumpy Road'
    elif classNo == 23:
        return 'Slippery Road'
    elif classNo == 24:
        return 'Road Narrows On The Right'
    elif classNo == 25:
        return 'Road Work'
    elif classNo == 26:
        return 'Traffic Signal'
    elif classNo == 27:
        return 'Pedestrians'
    elif classNo == 28:
        return 'Children Crossing'
    elif classNo == 29:
        return 'Bicycles Crossing'
    elif classNo == 30:
        return 'Beware Of Ice/Snow'
    elif classNo == 31:
        return 'Wild Animals Crossing'
    elif classNo == 32:
        return 'End Of All Speed And Passing Limits'
    elif classNo == 33:
        return 'Turn Right Ahead'
    elif classNo == 34:
        return 'Turn Left Ahead'
    elif classNo == 35:
        return 'Ahead Only'
    elif classNo == 36:
        return 'Go Straight Or Right'
    elif classNo == 37:
        return 'Go Straight Or Left'
    elif classNo == 38:
        return 'Keep Right'
    elif classNo == 39:
        return 'Keep Left'
    elif classNo == 40:
        return 'Roindabout Mandatory'
    elif classNo == 41:
        return 'End Of No Passing'
    elif classNo == 42:
        return 'End Of No Passing Of Vehicles Over 3.5 Metric Tons'

while True:
    #Read The Image
    success, imgOriginal = cap.read()

    #Process The Image
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow('Processed Image', img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(imgOriginal, 'Class: ', (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOriginal, 'Probability: ', (20,75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    #Prediction
    predictions = model.predict(img)
    classIndex = model.predict_classes(img)
    probabilityValue = np.amax(predictions)
    if probabilityValue > threshold:
        # print(getClassName(classIndex))
        cv2.putText(imgOriginal, str(classIndex)+ ' '+str(getClassName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOriginal, str(round(probabilityValue*100, 2))+ '%', (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('Result', imgOriginal)
    if cv2.waitKey(1) == 13:
        break