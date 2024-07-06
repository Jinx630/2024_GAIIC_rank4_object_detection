"""
@Time: 2024/5/10 23:14
@Author: xujinlingbj
@File: aug_utils.py
"""
import copy
import random
import sys

import cv2
import numpy as np

def swap_triangular_pixels(img):
    height, width = img.shape[:2]
    # 创建一个相同大小的临时图像，用于存放交换后的像素
    res_img = np.zeros_like(img)

    for y in range(height):
        for x in range(width):
            new_x = width - x - 1
            new_y = height - y - 1
            res_img[new_y, new_x] = img[y, x]

    return res_img

def swap_bbox(bbox):
    # x y w h
    bbox[0] = 640 - bbox[0] - bbox[2]
    bbox[1] = 512 - bbox[1] - bbox[3]

    return bbox

def highlight_image_edges(image):
    """边缘增强算法"""
    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用Canny算子进行边缘检测
    edges = cv2.Canny(gray_image, 100, 200)

    # 将边缘信息添加到原始图像上，使用黑色填充边缘
    image_with_edges = image.copy()  # 复制原图像用于添加边缘信息
    image_with_edges[edges != 0] = [0, 0, 0]  # 设置边缘为黑色
    return image_with_edges


def adjust_brightness_and_contrast(img, brightness=-150, contrast=1):
    """

    调整图片的亮度和对比度
    brightness: 亮度调节值，可以是负数到正数
    contrast: 对比度调节值，可以是负数到正数
    """

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(img, alpha_b, img, 0, gamma_b)
    else:
        buf = img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def add_gaussian_noise(image, mean=50, var=50):
    """给图像添加高斯噪声
    mean: 噪声的平均值
    var: 噪声的方差
    """
    row, col, ch = image.shape
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return noisy.clip(0, 255).astype(np.uint8)


def add_salt_pepper_noise(image, salt=2, pepper=0.05):
    """给图像添加椒盐噪声
    salt: 盐噪声比例
    pepper: 椒噪声比例
    """
    row, col, ch = image.shape
    s_vs_p = 0.5
    out = image.copy()
    # 盐噪声
    num_salt = np.ceil(row * col * salt)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[tuple(coords)] = 1

    # 椒噪声
    num_pepper = np.ceil(row * col * pepper)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[tuple(coords)] = 0
    return out


def crop_and_pad_image(img, dx):
    # dx 为裁剪/填充的像素数，dx[0] 是宽度方向，dx[1] 是高度方向
    dx = [int(x) for x in dx]
    h, w = img.shape[:2]

    if dx[0] > 0:
        # 裁剪图片左边
        img = img[:, dx[0] :]
        # 创建黑色填充区域（w、h、通道数）
        right_pad = np.zeros((h, dx[0], img.shape[2]), dtype=img.dtype)
        img = np.hstack((img, right_pad))
    elif dx[0] < 0:
        # 裁剪图片右边
        img = img[:, : dx[0]]
        # 创建黑色填充区域
        left_pad = np.zeros((h, abs(dx[0]), img.shape[2]), dtype=img.dtype)
        img = np.hstack((left_pad, img))

    # 更新尺寸因为可能已被修改
    h, w = img.shape[:2]

    if dx[1] > 0:
        # 裁剪图片上边
        img = img[dx[1] :, :]
        # 创建黑色填充区域
        down_pad = np.zeros((dx[1], w, img.shape[2]), dtype=img.dtype)
        img = np.vstack((img, down_pad))
    elif dx[1] < 0:
        # 裁剪图片下边
        img = img[: dx[1], :]
        # 创建黑色填充区域
        up_pad = np.zeros((abs(dx[1]), w, img.shape[2]), dtype=img.dtype)
        img = np.vstack((up_pad, img))

    return img


def rotate_image(image, angle):
    """
    旋转图像。
    :param image: 要旋转的图像。
    :param angle: 旋转角度。
    :return: 旋转后的图像。
    """
    # 获取图像尺寸和旋转中心
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # 计算旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    # 执行旋转
    rotated_image = cv2.warpAffine(image, M, (w, h))
    return rotated_image


class BlurImage(object):
    def __init__(self):
        super(BlurImage, self).__init__()
        self.functions = [
            self.gaussian_blur,
            self.mean_blur,
            self.median_blur,
            self.bilateral_filter_blur,
            self.motion_blur,
            self.resize_blur,
        ]
        self.index = random.randint(0, len(self.functions) - 1)

    def resize_blur(self, image):
        # 将图像分辨率缩小到其原始大小的一半
        # 使用cv2.resize，并将目标大小设置为原始大小的一半
        lower_res = cv2.resize(image, (0, 0), fx=0.2, fy=0.2)

        # 再将图像大小放大回原始大小
        # 注意，此时因为缩小再放大，图像会失去部分细节，从而产生模糊效果
        blurred_image = cv2.resize(lower_res, (image.shape[1], image.shape[0]))
        return blurred_image

    def gaussian_blur(self, image):
        # 应用高斯模糊
        blurred_image = cv2.GaussianBlur(image, (11, 11), 3)
        return blurred_image

    def mean_blur(self, image):
        # 5 一点模糊，15非常模糊
        blurred_image = cv2.blur(image, (10, 10))
        return blurred_image

    def median_blur(self, image):
        # 应用中值模糊
        blurred_image = cv2.medianBlur(image, 9)
        return blurred_image

    def bilateral_filter_blur(self, image):
        # 应用双边滤波

        blurred_image = cv2.bilateralFilter(image, 40, 45, 45)
        return blurred_image

    def motion_blur(self, image):
        # 20-30
        # 创建一个运动模糊核
        size = 25
        # size = random.randint(20, 30)
        kernel = np.zeros((size, size))
        kernel[int((size - 1) / 2), :] = np.ones(size)
        kernel = kernel / size

        # 应用运动模糊
        blurred_image = cv2.filter2D(image, -1, kernel)
        return blurred_image

    def forward(self, image):
        return self.functions[self.index](image)


def crop_boxes(bboxes, dx, img_w=640, img_h=512):
    """
    根据图片的裁剪/填充操作更新box列表。

    Args:
    - boxes: 列表，元素为[x,y,w,h]形式，表示box的左上角坐标和尺寸。
    - dx:    裁剪/填充的像素数，dx[0] 为宽度方向的修改量，dx[1] 为高度方向的修改量。

    Returns:
    - updated_boxes: 更新后的box列表。
    """
    updated_boxes = []
    for cls, center_x, center_y, w, h in bboxes:
        center_x = center_x * img_w
        center_y = center_y * img_h
        w = w * img_w
        h = h * img_h
        x = center_x - w / 2
        y = center_y - h / 2

        # 宽度方向裁剪/填充处理
        if dx[0] > 0:
            # 裁剪图像左侧
            x = max(0, x - dx[0])
        elif dx[0] < 0:
            # 向图像左侧填充
            x = min(x - dx[0], img_w)  # 直接偏移，因为左侧填充等同于向右移动box

        # 高度方向裁剪/填充处理
        if dx[1] > 0:
            # 裁剪图像顶部
            y = max(0, y - dx[1])
        elif dx[1] < 0:
            # 向图像顶部填充
            y = min(y - dx[1], img_h)  # 直接偏移，因为顶部填充等同于向下移动box
        new_center_x = (x + w / 2) / img_w
        new_center_y = (y + h / 2) / img_h
        new_w = w / img_w
        new_h = h / img_h
        updated_boxes.append([cls, new_center_x, new_center_y, new_w, new_h])  # 添加更新后的box

    return updated_boxes


def rotate_bbox(bboxes, angle, img_w=640, img_h=512):
    """
    旋转标注框坐标。
    :param bboxes: 原始标注框列表，每个元素为(label,  center_x, center_y, w, h)。
    :param angle: 旋转角度。
    :param img_w: 图像宽度。
    :param img_h: 图像高度。
    :return: 旋转后的标注框坐标列表。
    """

    # 计算旋转矩阵
    center = (img_w / 2, img_h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    rotated_bboxes = []
    for cls, center_x, center_y, w, h in bboxes:
        center_x = center_x * img_w
        center_y = center_y * img_h
        w = w * img_w
        h = h * img_h
        x = center_x - w / 2
        y = center_y - h / 2
        # 对标注框的四个顶点坐标执行旋转
        rect = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
        ones = np.ones(shape=(len(rect), 1))
        points_ones = np.hstack([rect, ones])

        # 应用旋转矩阵
        transformed_points = M.dot(points_ones.T).T

        # 计算旋转后的标注框坐标
        x0, y0 = np.min(transformed_points, axis=0)[:2]
        x1, y1 = np.max(transformed_points, axis=0)[:2]
        new_center_x = (x0 + x1) / 2 / img_w
        new_center_y = (y0 + y1) / 2 / img_h
        new_w = (x1 - x0) / img_w
        new_h = (y1 - y0) / img_h
        rotated_bboxes.append([cls, new_center_x, new_center_y, new_w, new_h])

    return rotated_bboxes


def rotate_and_crop_rgb_image(img):
    rand_xy = [10,9]+[8]*2+[7]*3+[6]*4+[5]*5+[4]*7+[3]*9+[2]*10+[1]*11 + [-10,-9]+[-8]*2+[-7]*3+[-6]*4+[-5]*5+[-4]*7+[-3]*9+[-2]*10+[-1]*11
    rand_a = [6,5]*1+[4]*2+[3]*4+[2]*12+[1]*40 + [-6,-5]*1+[-4]*2+[-3]*4+[-2]*12+[-1]*40
    enhancement_decision = random.randint(1, 3)  # 随机决定执行哪一种增强
    if enhancement_decision == 1:
        crop_x = random.randint(-10, 10)
        crop_y = random.randint(-10, 10)
        img = crop_and_pad_image(img, [crop_x, crop_y])
    elif enhancement_decision == 2:
        angle = random.uniform(-5, 5)
        img = rotate_image(img, angle)
    elif enhancement_decision == 3:
        crop_x = random.randint(-10, 10)
        crop_y = random.randint(-10, 10)
        img = crop_and_pad_image(img, [crop_x, crop_y])
        angle = random.uniform(-5, 5)
        img = rotate_image(img, angle)
    return img


def rotate_and_crop_tir_image(img, bbox):

    enhancement_decision = random.randint(1, 2)
    if enhancement_decision == 1:
        crop_x = random.randint(-10, 10)
        crop_y = random.randint(-10, 10)
        img = crop_and_pad_image(img, [crop_x, crop_y])
        bbox = crop_boxes(bbox, [crop_x, crop_y])
    elif enhancement_decision == 2:
        angle = random.uniform(-5, 5)
        img = rotate_image(img, angle)
        bbox = rotate_bbox(bbox, angle)
    elif enhancement_decision == 3:
        crop_x = random.randint(-10, 10)
        crop_y = random.randint(-10, 10)
        img = crop_and_pad_image(img, [crop_x, crop_y])
        bbox = crop_boxes(bbox, [crop_x, crop_y])
        angle = random.uniform(-5, 5)
        img = rotate_image(img, angle)
        bbox = rotate_bbox(bbox, angle)
    bbox = np.array(bbox)

    return img, bbox
