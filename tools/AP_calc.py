import os
import numpy as np
import xml.etree.ElementTree as ET


def IOU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


fileObject = open("list_car.txt", "r")
iou = {}
for im_no, line in enumerate(fileObject):
    #Read XML data for ground truth
    boxA =[0] * 4
    fileXML = line[0:6]+".xml"
    XMLPath = "/home/aqeel/Work/Datasets/VOCdevkit/VOC2007/Annotations/" + fileXML
    tree = ET.parse(XMLPath)
    root = tree.getroot()
    indexA=0
    print line[0:6]
    print len(root.findall('object'))
   # for child in root:
    for child in root.findall('object'):
        print child[0].text
        if child[0].text == 'car':
            indexA = indexA+1
            bb = child[4]
            boxA[0] = int(bb[0].text)
            boxA[1] = int(bb[1].text)
            boxA[2] = int(bb[2].text)
            boxA[3] = int(bb[3].text)

            for indexB,linedetect in enumerate(open("/home/aqeel/Work/py-faster-rcnn/tools/detected_cars.txt")):
                if linedetect[0:6]==line[0:6]:
                    boxB = [0] * 4
                    splitB = linedetect.split(",")
                    boxB = [int(splitB[1]), int(splitB[2]), int(splitB[3]), int(splitB[4])]
                    iou[indexA,indexB]=IOU(boxA, boxB)
                    zz=1
            zz=1
                   # if IOU(boxA, boxB) > 0.5:





