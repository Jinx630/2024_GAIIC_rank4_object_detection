"""
@Time: 2024/5/10 23:14
@Author: xujinlingbj
@File: aug_utils.py
"""
import copy
import random
import sys
import imgaug.augmenters as iaa
import cv2
import numpy as np
from PIL import Image, ImageEnhance


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


class AugColor(object):
    def __init__(self):
        super(AugColor, self).__init__()

        self.functions = [
            self.adjust_hue_saturation,
            self.iaacolor_augmentation_cv2,
        ]
        self.index = random.randint(0, len(self.functions) - 1)

    def adjust_hue_saturation(self, image, hue_delta=20, saturation_scale=2):
        """
        调整图像的色相和饱和度。

        参数:
        image：输入图像
        hue_delta：色相调整值，范围-180到180之间
        saturation_scale：饱和度缩放因子

        返回:
        调整色相和饱和度后的图像
        """
        # 将图像从BGR转换到HSV色彩空间
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float64)

        # 调整色相值
        hue, saturation, value = cv2.split(hsv_image)
        hue = np.mod(hue + hue_delta, 180).astype(np.float64)

        # 调整饱和度
        saturation = np.clip(saturation * saturation_scale, 0, 255).astype(np.float64)

        # 合并通道
        adjusted_hsv = cv2.merge([hue, saturation, value])

        # 将图像从HSV转换回BGR
        adjusted_image = cv2.cvtColor(adjusted_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return adjusted_image

    def iaacolor_augmentation_cv2(self, image):
        """它通过改变图片的颜色属性（如亮度、色相、饱和度等）来增强图像数据"""
        # 定义增强方法
        augmenter = iaa.Sequential([
            iaa.WithChannels(0, iaa.Add((10, 100))),  # 随机改变B通道
            iaa.WithChannels(1, iaa.Add((10, 100))),  # 随机改变G通道
            iaa.WithChannels(2, iaa.Add((10, 100)))  # 随机改变R通道
        ])

        # 将BGR图像转换为RGB图像，因为imgaug默认期望图像是RGB格式
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 应用增强
        augmented_image_rgb = augmenter(image=image_rgb)

        # 将增强后的RGB图像转回BGR格式以便使用cv2显示和保存
        augmented_image = cv2.cvtColor(augmented_image_rgb, cv2.COLOR_RGB2BGR)
        return augmented_image

    def forward(self, image):
        return self.functions[self.index](image)


class AugEdge(object):
    def __init__(self):
        super(AugEdge, self).__init__()
        self.functions = [
            self.apply_elastic_transformation,
            self.random_sharpness_enhancement,
            self.highlight_image_edges,
        ]
        self.index = random.randint(0, len(self.functions) - 1)

    def apply_elastic_transformation(self, image, alpha=20, sigma=5, random_state=None):
        """
        对图像应用弹性变形增强。

        参数:
        - image_path: 图像文件路径。
        - alpha: 弹性变形的强度。
        - sigma: 弹性变形场的平滑度。
        - random_state: 随机状态或种子。

        返回:
        - 增强后的图像。
        """
        # 确保图像在RGB颜色空间
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 定义弹性变形增强
        augmenter = iaa.ElasticTransformation(alpha=alpha, sigma=sigma, random_state=random_state)

        # 应用增强
        augmented_image = augmenter(image=image_rgb)

        # 将增强后的图像从RGB转换回BGR以用于cv2显示和保存
        augmented_image_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)

        return augmented_image_bgr

    def random_sharpness_enhancement(self, img_cv2):
        """随机锐度"""
        # 将cv2读取的图像（BGR格式）转换为PIL图像（RGB格式）
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_cv2)

        # 生成一个随机的锐度因子，这里假设范围在0.5到2.0之间
        # 1.0将保持原始图像不变，<1.0将减少锐度，>1.0将增加锐度
        # sharpness_factor = random.uniform(0.5, 2.0)
        sharpness_factor = -10
        # 创建一个锐度增强对象并应用随机的锐度
        enhancer = ImageEnhance.Sharpness(img_pil)
        img_sharpened = enhancer.enhance(sharpness_factor)

        # 将增强后的PIL图像转换回OpenCV格式以便后续处理或保存
        img_sharpened_cv2 = cv2.cvtColor(np.array(img_sharpened), cv2.COLOR_RGB2BGR)

        return img_sharpened_cv2

    def highlight_image_edges(self, image):
        """边缘增强算法"""
        # 转换为灰度图像
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 应用Canny算子进行边缘检测
        edges = cv2.Canny(gray_image, 100, 200)

        # 将边缘信息添加到原始图像上，使用黑色填充边缘
        image_with_edges = image.copy()  # 复制原图像用于添加边缘信息
        image_with_edges[edges != 0] = [0, 0, 0]  # 设置边缘为黑色
        return image_with_edges

    def forward(self, image):
        return self.functions[self.index](image)


class AugSaltNoise(object):
    def __init__(self):
        super(AugSaltNoise, self).__init__()
        self.functions = [
            self.add_speckle_noise,
            self.add_salt_pepper_noise,
        ]
        self.index = random.randint(0, len(self.functions) - 1)

    def add_speckle_noise(self, image, mean=0.5, std=0.5):
        """

        向图像添加Speckle噪声

        参数:
        image: 原始图像
        mean: 高斯噪声的均值，默认为0
        std: 高斯噪声的标准差，默认为0.1

        返回:
        带有Speckle噪声的图像
        """
        row, col, ch = image.shape
        gauss = np.random.normal(mean, std, (row, col, ch))
        # 生成Speckle噪声
        speckle = image * gauss
        noisy = image + speckle
        # 将图像值裁剪到0到255范围内
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy

    def add_salt_pepper_noise(self, image, salt=2, pepper=0.05):
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

    def forward(self, image):
        return self.functions[self.index](image)
