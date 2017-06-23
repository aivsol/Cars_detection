#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import math


global CLASSES
global person_bbox
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


def vis_detections(im, image_name, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    global person_bbox
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(im, aspect='equal')
    for i in inds:
        if class_name=='person':
            bbox = dets[i, :4]
            score = dets[i, -1]
            person_bbox=bbox
            ax.add_patch(
                 plt.Rectangle((bbox[0], bbox[1]),
                               bbox[2] - bbox[0],
                               bbox[3] - bbox[1], fill=False,
                               edgecolor='red', linewidth=3.5)
                 )
            ax.text(bbox[0], bbox[1] - 2,
                     '{:s} {:.3f}'.format(class_name, score),
                     bbox=dict(facecolor='blue', alpha=0.5),
                     fontsize=14, color='white')
            #Write detection parameters to text file to save it
            class_id = CLASSES.index(class_name)

            # if class_name == 'motorbike':
            #     file.write(image_name[-10:-1] +image_name[-1]+ ", " + repr(int(bbox[0])) + ", " + repr(int(bbox[1])) + ", " + repr(int(bbox[2])) + ", " + repr(int(bbox[3])) + ", " + str(class_id) + "\n")

            ax.set_title(('{} detections with ' + 'p({} | box) >= {:.1f}').format(class_name, class_name, thresh),fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    # im_file = "/home/aqeel/Work/Datasets/VOCdevkit/VOC2007/JPEGImages/" + image_name
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im,image_name, cls, dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
   # parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
    #                    default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

def shortlist_images(class_name):
    path_imagesets = "/home/aqeel/Work/Datasets/VOCdevkit/VOC2007/ImageSets/Main/" + class_name + "_test.txt"
    list_cars=[]
    fileCar = open("list_" + class_name + ".txt", "w")
    with open(path_imagesets) as fp:
        for line in fp:
            z=int(line[-3:-1])
            if z==1:
                list_cars = list_cars + [line[0:6]+ ".jpg"]
                fileCar.write(line[0:6]+"\n")
    fileCar.close()
    return list_cars

def dist_obj(bbox_L, bbox_R):
    L_mid = person_L[0] + (person_L[2] - person_L[0]) / 2
    R_mid = person_R[0] + (person_R[2] - person_R[0]) / 2

    disparity = L_mid - R_mid
    #camera parameteres
    B = 0.16
    X = 2048
    theta = 69.4
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print "Finding Distance ...  "
    D = (B * X) / (math.tan(theta / 2 * math.pi / 180) * disparity)
    print "The object distance is", D , "meters"

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    # list_cars=shortlist_images("car")
    # list_person=shortlist_images("person")
    # list_motorbike=shortlist_images("motorbike")

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

   # if args.cpu_mode:
        caffe.set_mode_cpu()
    #else:
     #   caffe.set_mode_gpu()http://host.robots.ox.ac.uk/pascal/VOC/voc2007/#testdata
      #  caffe.set_device(args.gpu_id)
       # cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)
    #im_names = list_motorbike
    # im_names = ['L.jpg', 'R.jpg']
   # for im_name in im_names:
   #  for im_name in im_names:
    im_nameL = "L.jpg"
    im_nameR = "R.jpg"
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print "Processing image name: ", im_nameL
    print 'Demo for data/demo/{}'.format(im_nameL)
    demo(net, im_nameL)
    person_L = person_bbox

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print "Processing image name: ", im_nameR
    print 'Demo for data/demo/{}'.format(im_nameR)
    demo(net, im_nameR)
    person_R = person_bbox
    dist_obj(person_L, person_R)


  #  file.close()
    plt.show()

