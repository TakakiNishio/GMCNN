# python library
import os
import getpass
import matplotlib.pyplot as plt
import numpy as np
import imghdr # image file type checker
import random
import argparse

# OpenCV
import cv2


# save list as .txt data
def save_list(label_list,file_address_name):

    f = open(file_address_name,'w')
    for label in label_list:
        f.write(str(label) + '\n')


# increase dataset for test
def increase(image_path_list, label_list, output_image_path, output_txt_path):

    data_N = len(image_path_list)
    print data_N
    print len(label_list)

    im_cnt = 0

    increased_image_path = []
    increased_label = []

    for i in range(data_N):

        # original image
        origin_img = cv2.imread(image_path_list[i])
        label = label_list[i]

        image_path = output_image_path+'/'+str(im_cnt)+'.jpg'
        increased_image_path.append(image_path)
        increased_label.append(label)
        cv2.imwrite(image_path, origin_img)
        im_cnt += 1

        # flip horizontally
        flipped_img = cv2.flip(origin_img, 1)
        image_path = output_image_path+'/'+str(im_cnt)+'.jpg'
        increased_image_path.append(image_path)
        increased_label.append(label)
        cv2.imwrite(image_path, flipped_img)
        im_cnt += 1

        if i % 10 == 0:
            print str(i) + '/' + str(data_N)

    save_list(increased_image_path, output_txt_path+'/images.txt')
    save_list(increased_label, output_txt_path+'/labels.txt')

    print 'total image: ' + str(im_cnt)


# main
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='extract object images')
    parser.add_argument('--input_txt_directory', '-itd', type=str, default="data",help='input txt path')
    parser.add_argument('--output_txt_directory', '-otd', type=str, default="data",help='output txt path')
    parser.add_argument('--output_image_directory', '-oid', type=str, default=False,help='output image path')
    args = parser.parse_args()

    dataset_path = args.input_txt_directory

    image_path_txt = open(dataset_path+'/images.txt','r')
    label_txt = open(dataset_path+'/labels.txt','r')

    image_path_list = []
    label_list = []

    for image_path in image_path_txt:
        image_path_list.append(image_path.split('\n')[0])

    for label in label_txt:
        label_list.append(label.split('\n')[0])

    output_image_path = args.output_image_directory
    output_txt_path = args.output_txt_directory

    if not output_image_path is False and not os.path.isdir(output_image_path):
        os.makedirs(output_image_path)

    if not output_txt_path is False and not os.path.isdir(output_txt_path):
        os.makedirs(output_txt_path)

    print 'directory: ' + dataset_path

    increase(image_path_list, label_list, output_image_path, output_txt_path)
 
