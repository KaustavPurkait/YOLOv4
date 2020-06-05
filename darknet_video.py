from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
from collections import defaultdict
import  distance

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img,colour = None):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        if colour == 'red':
            cv2.rectangle(img, pt1, pt2, (255, 0, 0), 1)
        else:
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
##        cv2.putText(img,
##                    detection[0].decode() +
##                    " [" + str(round(detection[1] * 100, 2)) + "]",
##                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
##                    [0, 255, 0], 2)
    return img


netMain = None
metaMain = None
altNames = None


def YOLO():

    global metaMain, netMain, altNames
    configPath = "./cfg/yolov4.cfg"
    weightPath = "./yolov4.weights"
    metaPath = "./cfg/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("test.mp4")
    cap.set(3, 1280)
    cap.set(4, 720)

##    out = cv2.VideoWriter(
##        "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30,
##        (darknet.network_width(netMain), darknet.network_height(netMain)))
    
    out = cv2.VideoWriter(
        "output.mp4", cv2.VideoWriter_fourcc(*"MP4V"), 30,
        (1280, 720))
    
    print("Starting the YOLO loop...")
    print("width =",darknet.network_width(netMain), "height=", darknet.network_height(netMain))
    
    ret, frame_read = cap.read()
    while ret:
        prev_time = time.time()
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)
        
        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.35)
        detections  = [i for i in detections if i[0].decode('ASCII')=='person']
        
        #print(detections)
        violations,non_violations = distance.calc_distance(detections,darknet.network_height(netMain))

        image = cvDrawBoxes(violations, frame_resized, colour = "red")
        image = cvDrawBoxes(non_violations, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(1280,720), interpolation=cv2.INTER_LINEAR)

        out.write(image)
        
        print(1/(time.time()-prev_time))
        
        ret, frame_read = cap.read()
#        cv2.imwrite('../image.jpg', image)
#        break
    cap.release()
    out.release()

if __name__ == "__main__":
    YOLO()
