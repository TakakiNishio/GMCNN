# python library
import os
import getpass
import matplotlib.pyplot as plt
import numpy as np
import imghdr # image file type checker
import random
import argparse
import os


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

    parser = argparse.ArgumentParser(description='prepare txt data for training')
    parser.add_argument('--negative_data_directory', '-nd', type=str, default=False,help='negative data path')
    parser.add_argument('--positive_data_directory', '-pd', type=str, default=False,help='positive data path')
    parser.add_argument('--output_txt_path', '-od', type=str, default="data", help='output path')
    args = parser.parse_args()

    if not os.path.isdir(args.output_txt_path):
        os.makedirs(args.output_txt_path)

    # negative dataset
    negative_images = []
    negative_labels = []
    for f in find_all_files(args.negative_data_directory):
        if os.path.isfile(f):
            negative_images.append(f)
            negative_labels.append(0)
    
    # positive dataset
    positive_images = []
    positive_labels = []
    for f in find_all_files(args.positive_data_directory):
        if os.path.isfile(f):
            positive_images.append(f)
            positive_labels.append(1)

    # total dataset
    total_images = negative_images + positive_images
    total_labels = negative_labels + positive_labels

    print "negative data: "+str(len(negative_images))
    print "positive data: "+str(len(positive_images))
    print "total data: "+str(len(total_images))

    save_list(total_images, args.output_txt_path+'/images.txt')
    save_list(total_labels, args.output_txt_path+'/labels.txt')

    print "txt data is saved."
