# -*- coding: utf-8 -*-

# Python libraries
import os
import numpy as np
from numpy.random import *
from matplotlib import pylab as plt
import random
import argparse

# OpenCV
import cv2


# main
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='prepare txt data for training')
    parser.add_argument('--txt_directory', '-d', type=str, default="data",help='negative data path')
    parser.add_argument('--testdata', '-testdata', action="store_true")
    args = parser.parse_args()

    dataset_path = args.txt_directory
    end_flag = False
    fontType = cv2.FONT_HERSHEY_SIMPLEX

    image_path_txt = open(dataset_path+'/images.txt','r')
    label_txt = open(dataset_path+'/labels.txt','r')

    image_path_list = []
    label_list = []

    for image_path in image_path_txt:
        image_path_list.append(image_path.split('\n')[0])

    for label in label_txt:
        if args.testdata:
            label_list.append(label.split('\n')[0])
        else:
            label_list.append(int(label))

    data_N = len(image_path_list)

    print data_N
    print len(label_list)

    while(True):

        random_index = random.randint(0, data_N-1)
        print str(image_path_list[random_index])

        img = cv2.imread(image_path_list[random_index])
        label = label_list[random_index]

        height = img.shape[0]
        width = img.shape[1]

        if args.testdata:
            msg = label
            color = (0,255,0)
        else:
            if label == 1:
                msg = "POSITIVE"
                color = (221,178,68)
            else:
                msg = "NEGATIVE"
                color = (250,53,225)

        print msg
        cv2.putText(img, "label: "+msg, (5, height-20), fontType, 0.8, color, 2)
        cv2.imshow("image", img)

        key = cv2.waitKey(2000) & 0xFF

        if key & 0xFF == 27:
            break


        # while (True):

        #     key = cv2.waitKey(1) & 0xFF

        #     # load next image
        #     if key & 0xFF == ord("z"):
        #         break

        #     # finish this program
        #     if key & 0xFF == 27:
        #         end_flag = True
        #         break

        # if end_flag == True:
        #     break

    cv2.destroyAllWindows()
