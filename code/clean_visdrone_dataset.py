"""
@Time: 2024/5/8 09:25
@Author: xujinlingbj
@File: clean_visdrone_dataset.py
"""
import argparse
import json
import math
import os
import shutil
import sys
from collections import defaultdict
import cv2
import pandas as pd
from tqdm import tqdm

from process_data.utils import crop_image_and_bbox, change_image2gray

"""
[{(1400, 1050): 2646, (1400, 788): 2180, (1360, 765): 1273, (2000, 1500): 756, (1920, 1080): 384, (960, 540): 334, (1916, 1078): 670, (1344, 756): 1, (1398, 1048): 30, (480, 360): 1})

"""
id2label = {
    0: "ignored regions",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",  # 三轮车
    8: "awning-tricycle", # 遮阳篷三轮车
    9: "bus",
    10: "motor",
    11: "others",
}

save_id2label = {0: "car", 1: "truck", 2: "bus", 3: "van", 4: "freight_car"}

choose_category = ['car', 'van', 'trunk', 'bus']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_annotation_data_dir', type=str,
                        default='/Users/xujinlingbj/pycharmWork/game/gaicc2024/dataset/VisDrone/VisDrone2019-DET-train/annotations',
                        help='initial weights path')
    parser.add_argument('--rgb_image_save_dir', type=str,
                        default='/Users/xujinlingbj/pycharmWork/game/gaicc2024/dataset/VisDrone/VisDrone_yolo_format/rgb',
                        help='initial weights path')

    opt = parser.parse_args()

    train_annotation_data_dir = opt.train_annotation_data_dir
    val_annotation_data_dir = train_annotation_data_dir.replace("train", "val")
    test_annotation_data_dir = train_annotation_data_dir.replace("train", "test")
    image_save_dir = opt.rgb_image_save_dir
    tir_image_save_dir = image_save_dir.replace('rgb', 'tir')
    annotation_save_dir = image_save_dir.replace('rgb', 'labels')
    for path in [image_save_dir, annotation_save_dir, tir_image_save_dir]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
    cnt = 0
    save_label2id = {v: k for k, v in save_id2label.items()}

    save_number = 0
    image_shape_list = defaultdict(int)
    for data_dir in [train_annotation_data_dir, val_annotation_data_dir, test_annotation_data_dir]:

        print(data_dir)
        image_dir = data_dir.replace('annotations', 'images')
        cate_ids = defaultdict(int)
        files = os.listdir(data_dir)
        all_data = []
        for file in tqdm(files, total=len(files), desc='clean: '):
            image_file_name = file.split('.')[0] + '.jpg'

            with open(os.path.join(data_dir, file), "r") as f:
                annotation_data = f.readlines()
            bad_case_flag = False
            choose_cate_dict = defaultdict(int)
            choose_bbox_list = []
            show_flag = False
            image = cv2.imread(os.path.join(image_dir, image_file_name))
            for line in annotation_data:
                line = line.strip().split(",")
                # no truncation = 0 (truncation ratio 0%), and partial truncation = 1 (truncation ratio 1% ~ 50%)
                truncation = int(line[6])
                # no occlusion = 0 (occlusion ratio 0%),
                # partial occlusion = 1 (occlusion ratio 1% ~ 50%),
                # and heavy occlusion = 2 (occlusion ratio 50% ~ 100%)
                occlusion = int(line[7])
                # if occlusion == 2:
                #     continue
                bbox = [int(x) for x in line[:4]]
                if int(line[5]) not in [4, 5, 6, 9]:
                    choose_cate_dict['no'] += 1
                    continue

                label = id2label[int(line[5])]
                choose_cate_dict['yes'] += 1
                if label not in save_label2id:
                    save_label2id[label] = len(save_label2id)
                choose_bbox_list.append([save_label2id[label]]+bbox)

            if not bad_case_flag and choose_cate_dict['yes'] > 0:

                image = cv2.imread(os.path.join(image_dir, image_file_name))
                origin_image_width, origin_image_height, _ = image.shape

                crop_width = 640
                crop_height = 512
                # New sliding window step sizes
                # x_step = 640-160
                # y_step = 512-128
                # x_start = 0
                # y_start = 0
                # Calculate the number of crops horizontally and vertically
                num_crops_horizontal = max(1, math.floor(origin_image_width / crop_width))
                num_crops_vertical = max(1, math.floor(origin_image_height / crop_height))

                overlap_horizontal = 0 if num_crops_horizontal * crop_width < origin_image_width else (num_crops_horizontal * crop_width - origin_image_width) // max(1, num_crops_horizontal - 1)
                overlap_vertical = 0 if num_crops_vertical * crop_height < origin_image_height else (num_crops_vertical * crop_height - origin_image_height) // max(1, num_crops_vertical - 1)
                for i in range(num_crops_horizontal):
                    for j in range(num_crops_vertical):
                        suffix = 'vis_drone_' + str(save_number) + '_'
                        save_number += 1
                        x_start = max(0, min(origin_image_width - crop_width, i * crop_width - i * overlap_horizontal))
                        y_start = max(0, min(origin_image_height - crop_height, j * crop_height - j * overlap_vertical))

                # for x_start in range(0, origin_image_width-crop_width+1, x_step):
                #     for y_start in range(0, origin_image_height-crop_height+1, y_step):
                #         suffix = 'vis_drone_' + str(save_number) + '_'
                #         save_number += 1
                        crop_image, cropped_yolo_label = crop_image_and_bbox(image, choose_bbox_list, crop_width=crop_width,
                                                                             crop_height=crop_height, crop_x_start_id=x_start,
                                                                             crop_y_start_id=y_start)
                        # 16287 去掉空白屏 13387
                        if len(cropped_yolo_label) == 0:
                            continue
                        image_height, image_width, _ = crop_image.shape
                        rgb_save_path = os.path.join(image_save_dir, suffix+image_file_name)
                        cv2.imwrite(rgb_save_path, crop_image)
                        gray_image = change_image2gray(rgb_save_path)
                        cv2.imwrite(os.path.join(tir_image_save_dir, suffix+image_file_name), gray_image)

                        with open(os.path.join(annotation_save_dir, suffix+file), 'w') as fw:
                            for line in cropped_yolo_label:
                                cx = (line[1] + line[3] / 2) / image_width
                                cy = (line[2] + line[4] / 2) / image_height
                                width = line[3] / image_width
                                height = line[4] / image_height
                                line = [line[0], cx, cy, width, height]
                                line = map(str, line)
                                fw.write(' '.join(line) + '\n')

        print(f'保存的图片个数：{len(os.listdir(image_save_dir))}, 保存的label个数：{len(os.listdir(annotation_save_dir))}')
        print(f'save_label2id={save_label2id}')
    final_save_id2label = {v:k for k, v in save_label2id.items()}
    print(list(save_label2id.keys()))
    with open(os.path.join('/'.join(image_save_dir.split('/')[:-1]), 'id2label.json'), 'w') as f:
        json.dump(final_save_id2label, f, ensure_ascii=False, indent=4)



