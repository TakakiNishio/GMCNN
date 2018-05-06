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


# high contrast and low contrast table
def cont():

    min_table = 5
    max_table = 210
    diff_table = max_table - min_table

    LUT_HC = np.arange(256, dtype = 'uint8' )
    LUT_LC = np.arange(256, dtype = 'uint8' )

    for i in range(0, min_table):
        LUT_HC[i] = 0
    for i in range(min_table, max_table):
        LUT_HC[i] = 255 * (i - min_table) / diff_table
    for i in range(max_table, 255):
        LUT_HC[i] = 255

    for i in range(256):
        LUT_LC[i] = min_table + i * (diff_table) / 255

    return LUT_HC, LUT_LC


# add gaussian noise
def noise(img):
    row,col,ch= img.shape
    mean = 2
    sigma = 8
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    gauss_img = img + gauss
    return gauss_img


# save label list
def save_label_list(label_list,file_address_name):

    f = open(file_address_name,'w')
    for label in label_list:
        f.write(str(label) + '\n')


# increase dataset
def increase(dataset_path, save_path, n):

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    LUT_HC, LUT_LC = cont()

    im_cnt = 0

    files = sorted(os.listdir(dataset_path))
    data_N = len(files)

    for i in range(0,data_N,n):

        # original image
        origin_img = cv2.imread(dataset_path+'/'+files[i])
        cv2.imwrite(save_path+'/'+str(im_cnt)+'.jpg', origin_img)
        im_cnt += 1

        if i % 100 == 0:
            print str(i) + '/' + str(data_N)

    print 'total image: ' + str(im_cnt)

# main
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='extract object images')
    parser.add_argument('--label', '-l', type=int, default=0, help='specified label')
    parser.add_argument('--data_n', '-n', type=int, default=10,help='input image')
    parser.add_argument('--input_image_directory', '-id', type=str, default=False,help='input image')
    parser.add_argument('--output_image_directory', '-od', type=str, default=False,help='output image')
    args = parser.parse_args()

    input_path = args.input_image_directory
    output_path = args.output_image_directory

    if not output_path is False and not os.path.isdir(output_path):
        os.makedirs(output_path)

    label = args.label # 0 or 1

    print 'label: ' + str(label)
    print 'directory: ' + input_path

    increase(input_path, output_path, args.data_n)
 
