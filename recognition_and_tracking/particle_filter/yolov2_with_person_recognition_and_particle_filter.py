#python libraries
import cv2
import numpy as np
import argparse
import os

#chainer
from chainer import serializers, Variable
import chainer.functions as F

#python scripts
from yolov2 import *
from CocoPredictor import *
from PersonClassifier import *
from network_structure import *
from particle_filter import *


#preparing to save the video
def initWriter(w, h, fps, save_path):
    fourcc = cv2.VideoWriter_fourcc('F','L','V','1')
    rec = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
    return rec


#main
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='yolov2_darknet_predict for video')
    parser.add_argument('--video_file', '-v', type=str, default=False,help='path to video')
    parser.add_argument('--camera_ID', '-c', type=int, default=0,help='camera ID')
    parser.add_argument('--save_name', '-s', type=str, default=False,help='camera ID')
    args = parser.parse_args()

    if not args.video_file == False:
        cap = cv2.VideoCapture(args.video_file)
    else:
        cap = cv2.VideoCapture(args.camera_ID)

    ret, frame = cap.read()
    height, width, channels = frame.shape

    rec = False
    if not args.save_name == False:

        save_path = 'results/videos/'

        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        rec = initWriter(width, height, 30, save_path+args.save_name)

    coco_predictor = CocoPredictor()

    model_path1 = 'models/person_classifier/thibault_model5/'
    model1 = CNN_thibault2()
    image_size1 = 50
    person_classifier = PersonClassifier(model_path1,model1,image_size1)

    th = 0.9

    camshift_scale = 0.1

    cv2.namedWindow("video", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("dominant", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("low", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("high", cv2.WINDOW_NORMAL)

    end_flag = False
    target_counter = 0
    target_center = (0,0)
    past_target_center = (0,0)

    image_size = (frame.shape[0], frame.shape[1])

    pf = ParticleFilter(image_size)
    pf.initialize()

    while(True):

        ret, frame = cap.read()

        if ret is not True or end_flag is True:
            break

        nms_results = coco_predictor(frame)

        for result in nms_results:
            left, top = result["box"].int_left_top()
            right, bottom = result["box"].int_right_bottom()
            color = (255, 0, 255)

            if result["class_id"] == 0:
                person_class, prob = person_classifier(frame[top:bottom, left:right])

                if person_class == 1:

                    if prob > th:

                        color = (0,255,0)
                        result_info = 'TARGET(%2d%%)' % (prob*100)

                        w = right-left
                        h = bottom-top

                        past_target_center = target_center
                        target_center = (left+int(w*0.5),top+int(h*0.5))

                        dist = np.linalg.norm(np.asarray(past_target_center)-np.asarray(target_center))

                        if dist < 50:

                            target_counter += 1

                            if target_counter < 5:
                                continue

                            crop_param_w = 3
                            crop_param_h = 4

                            person_top = target_center[1] - int(h/crop_param_h)
                            person_bottom = target_center[1]- int((h/crop_param_h)/2)
                            person_left = target_center[0]-int(w/crop_param_w)
                            person_right = target_center[0]+int(w/crop_param_w)
                            # person_left = target_center[1] - int(w/crop_param)
                            # person_right = target_center[1] + int(w/crop_param)

                            # person_left = target_center[0]-int(w*camshift_scale)
                            # person_top = target_center[1]-2*int(h*camshift_scale)
                            # person_right = person_left + int(w*camshift_scale)
                            # person_bottom = person_top + int(h*camshift_scale)

                            crop_center = (person_left+int((person_right-person_left)*0.5),
                                           person_top+int((person_bottom-person_top)*0.5))

                            cropped_img = frame[person_top:person_bottom, person_left:person_right]
                            # cropped_img = frame[top:bottom, left:right]
                            print(cropped_img.shape)

                            pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))

                            dominant_bgr = get_dominant_color(pil_img)

                            # convert BGR to HSV
                            dominant_hsv = bgr_to_hsv(dominant_bgr)

                            print("dominant bgr")
                            print(dominant_bgr)

                            print("dominant hsv")
                            print(dominant_hsv)
                            print()

                            s_range = 5
                            if dominant_hsv[0] < s_range:
                                low_s = 0
                            else:
                                low_s = dominant_hsv[0]-s_range

                            if dominant_hsv[0]+s_range > 179 :
                                high_s = dominant_hsv[0]
                            else:
                                high_s = dominant_hsv[0] + s_range

                            _LOWER_COLOR = np.array([low_s,50,50])
                            _UPPER_COLOR = np.array([high_s,255,255])

                            # _LOWER_COLOR = np.array([low_s,50,50])
                            # _UPPER_COLOR = np.array([high_s,180,180])

                            # _LOWER_COLOR = np.array([low_s,dominant_hsv[1]-40,dominant_hsv[2]-40])
                            # _UPPER_COLOR = np.array([high_s,dominant_hsv[1]+80,dominant_hsv[2]+80])

                            low_bgr = hsv_to_bgr(_LOWER_COLOR)
                            high_bgr = hsv_to_bgr(_UPPER_COLOR)

                            print("lower hsv")
                            print(_LOWER_COLOR)
                            print("upper hsv")
                            print(_UPPER_COLOR)
                            print()

                            # display doninant color
                            size = 200, 200, 3
                            dominant_color_display = np.zeros(size, dtype=np.uint8)
                            dominant_color_display[:] = dominant_bgr

                            # display modified dominant color
                            low_bgr_display = np.zeros(size, dtype=np.uint8)
                            low_bgr_display[:] = low_bgr

                            # display modified dominant color
                            high_bgr_display = np.zeros(size, dtype=np.uint8)
                            high_bgr_display[:] = high_bgr

                            # cv2.imshow("target", frame[top:bottom, left:right])
                            # cv2.imshow("cropped image", cropped_img)
                            # cv2.imshow("dominant", dominant_color_display)
                            # cv2.imshow("low", low_bgr_display)
                            # cv2.imshow("high", high_bgr_display)

                            RUN_PF(cap, rec, pf, _LOWER_COLOR, _UPPER_COLOR, dominant_bgr, high_bgr, crop_center)

                        cv2.circle(frame, target_center, 5, (0, 215, 253), -1)

                    else:
                        target_counter = 0
                        result_info = 'OTHERS(%2d%%)' % ((1.0-prob)*100)
                else:
                    result_info = 'OTHERS(%2d%%)' % ((1.0-prob)*100)

                #print(result_info)

                cv2.putText(frame, result_info, (left, bottom+25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        cv2.putText(frame, 'Searching with YOLOv2...', (10,18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.imshow("video", frame)

        if not args.save_name == False:
            rec.write(frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            end_flag = True

    cap.release()

    if not args.save_name == False:
        rec.release()
