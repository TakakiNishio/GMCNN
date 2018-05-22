# python library
import os
import getpass
import matplotlib.pyplot as plt
import numpy as np
import imghdr # image file type checker
import random
import argparse
import copy

# OpenCV
import cv2


# save list as .txt data
def save_list(label_list,file_address_name):

    f = open(file_address_name,'w')
    for label in label_list:
        f.write(str(label) + '\n')


# load image path
def find_all_files(directory):
    for root, dirs, files in os.walk(directory):
        yield root
        for file in files:
            yield os.path.join(root, file)


# main
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='prepare txt data for test')
    parser.add_argument('--negative_data_directory', '-nd', type=str, default=False,help='negative data path')
    parser.add_argument('--positive_data_directory', '-pd', type=str, default=False,help='positive data path')
    parser.add_argument('--mixed_data_directory', '-md', type=str, default=False,help='mixed data path')
    parser.add_argument('--output_txt_path', '-od', type=str, default="data", help='output path')
    args = parser.parse_args()

    print

    # negative dataset
    if not args.negative_data_directory is False:

        output_path = args.output_txt_path+"/negative"
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        print "Negative dataset preparation"
        negative_images = []
        negative_labels = []
        for f in find_all_files(args.negative_data_directory):
            if os.path.isfile(f):
                negative_images.append(f)
                negative_labels.append("10")

        save_list(negative_images, output_path+'/images.txt')
        save_list(negative_labels, output_txt_path+'/labels.txt')
        print "data_N: "+str(len(negative_images))
        print "txt data is saved."
        print

    
    # positive dataset
    if not args.positive_data_directory is False:

        output_path = args.output_txt_path+"/positive"
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        print "Positive dataset preparation"
        positive_images = []
        positive_labels = []
        for f in find_all_files(args.positive_data_directory):
            if os.path.isfile(f):
                positive_images.append(f)
                positive_labels.append("01")

        save_list(positive_images, output_path+'/images.txt')
        save_list(positive_labels, output_path+'/labels.txt')
        print "data_N: "+str(len(positive_images))
        print "txt data is saved."
        print


    # mixed dataset
    if not args.mixed_data_directory is False:

        output_path = args.output_txt_path+"/mixed"
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        print "Mixed dataset preparation"
        mixed_images = []
        mixed_labels = []
        for f in find_all_files(args.mixed_data_directory):
            if os.path.isfile(f):
                mixed_images.append(f)

        data_N = len(mixed_images)
        fontType = cv2.FONT_HERSHEY_SIMPLEX

        for i in range(data_N):

            msg_num = "image: " + str(i+1) + "/" + str(data_N) +" (type key Z to load next image)"
            msg_howto = "input the correct label (ex. A:2,B:1 -> 21)"
            print msg_num
            print mixed_images[i]
            print msg_howto

            img = cv2.imread(mixed_images[i])
            height = img.shape[0]

            cv2.putText(img, msg_num, (5, height-20), fontType, 0.7, (255,0,0), 2)
            cv2.putText(img, msg_howto, (5,25), fontType, 0.9, (255,0,0), 2)
            cv2.imshow("image", img)
            label = []
            label_flag = False
            copied_img = copy.deepcopy(img)

            while (True):

                cv2.imshow("image", img)
                key = cv2.waitKey(1) & 0xFF

                if not key == 255:
                    try:
                        num = int(chr(key).replace(' ',''))
                        print "n: "+str(num)
                        label.append(num)
                        img = copy.deepcopy(copied_img)
                        if len(label) == 2:
                            label_str = str(label[0])+str(label[1])
                            label = []
                            label_flag = True
                            msg = "label: "+label_str
                            cv2.putText(img, msg, (5,60), fontType, 0.9, (0,255,0), 2)
                            print msg

                    except ValueError:
                        # print "not a number"
                        pass

                # load next image
                if key == ord("m"):
                    if label_flag:
                        mixed_labels.append(label_str)
                        print "label: "+ label_str +" is saved."
                        print
                        break
                    else:
                        msg_error = "please input the correct label!!"
                        print msg_error
                        cv2.putText(img, msg_error, (5, height-40), fontType, 0.7, (0,0,255), 2)

        save_list(mixed_images, output_path+'/images.txt')
        save_list(mixed_labels, output_path+'/labels.txt')

        print "data_N: "+str(len(mixed_images))
        print "txt data is saved."
        print
