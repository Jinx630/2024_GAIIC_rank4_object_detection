## 训练，推理
### 训练：线上A榜单模型 0.5252，训练配置 4xA100，early stop at 50epoch，8 小时，外部数据 visdrone-det

```
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9516 code/train.py \
    --workers 8 \
    --device 0,1,2,3 \
    --batch 64 \
    --data code/data/detect_viscutmorewithoutobj.yaml \
    --img 640 \
    --epochs 1 \
    --cfg code/models/detect/gelan-e-double-img-mlab.yaml \
    --weights data/pretrain_model/gelan-e.pt \
    --project data/model_data/ckpt-001-4prelatermlab \
    --model_size e \
    --save_period 1 \
    --min-items 0 \
    --epochs 100 \
    --custom_aug true \
    --close-mosaic 0
```

    
### 推理：推理速度单卡A100 60张/s

```
python code/index.py 	# task = 'val' or 'test'
```


## 环境配置
参考 [YOLOv9](https://github.com/WongKinYiu/yolov9) 和 [ICAFusion](https://github.com/chanchanchan97/ICAFusion) 环境配置，衷心感谢这两个项目的作者。

## 数据
数据存放路径参照 code/data/detect_viscutmorewithoutobj.yaml 文件

注意：需要按照如下步骤将外部数据放置到指定位置。

### DroneVehicle 数据集
1、获取链接
https://github.com/VisDrone/DroneVehicle

2、下载并放置文件 \
（1）下载文件：Test \
（2）在官方 contest_data 目录下新建 droneVehicle 文件夹。\
（3）将上述3个文件夹放到 contest_data/droneVehicle 文件夹下。

3、具体处理方案 \
直接将全部的 Test 数据混合到比赛的训练集中。

### VisDrone-DET 数据集
1、获取链接
https://github.com/VisDrone/VisDrone-Dataset 

2、下载并放置文件 \
（1）需要下载的文件包括：trainset (1.44 GB)、valset (0.07 GB)、testset-dev (0.28 GB)。\
（2）在官方 contest_data 目录下新建 visdrone 文件夹。\
（3）将上述3个文件夹放到 contest_data/visdrone 文件夹下。

3、具体处理方案 \
（1）只保留比赛类别：'car', 'van', 'trunk', 'bus'。 \
（2）按照裁剪尺寸640x512，计算每张图的最大裁剪数量，并进行裁剪。 \
（3）去除裁剪后没有比赛类别的图。 \
（4）对每一张裁剪的 rgb 图，转换成一张灰度图，作为和 rgb 图像配对的 tir 图像。 \
（5）对此数据的trainset、valset、testset-dev执行上述4个步骤，最终可以得到 15929 对 rgb-tir 样本。

4、处理 vis_drone 数据

```
python code/clean_visdrone_dataset.py --train_annotation_data_dir data/contest_data/ --rgb_image_save_dir
```


## 预训练权重 放置在 data/pretrain_model
（1）gelan-e：https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-e.pt \
（2）yolov9-e：https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt 
 
## 算法

### 整体思路介绍
本赛题旨在通过使用无人机拍摄的可见光-红外图像对，构建目标检测模型，以识别出五类车辆的具体位置和类别信息，可归类为“双光目标检测”领域的挑战。

该赛题具备一系列独特的特性。首先，受拍摄场景和角度多样性的影响，数据范围涵盖城市街道、农村区域、居民区、停车场，以及从白天到夜晚的各种场景，
使得图片品质存在较大差异，且噪声问题显著。其次，测试集的数据采集时间与训练集不同，涉及多样的属性特征，如不同的天气条件、光照情况、采集场景等，
造成测试集与训练集分布的不一致性。因此，本赛题主要面临两大难点：一是如何在图片噪声较多的情况下提升模型的识别准确性；
二是如何增强模型在测试集上的鲁棒性。

为应对这些挑战，本项目构建基于改进的gelan双光目标检测模型。我们首先将单光gelan模型扩展为双光版本，并集成了ICAFusion模型的融合单元，
旨在有效融合双模态特征。进一步地，借鉴yolov9的策略，我们为双光gelan模型引入了四个辅助分支，以加强对浅层特征的监督，从而提升模型性能。
此外，为了增强模型对不同场景的识别能力，我们将与赛题场景相似的VisDrone数据集和DroneVehicle测试集并入训练数据，
考虑到VisDrone缺少红外图像，采用灰度图作为替代。最后，本项目在训练过程中采用多种数据增强技术，包括裁剪、旋转、模糊、亮度调整、
椒盐高斯噪声处理、边界增强等方法，旨在进一步提高模型在测试集上的泛化性。

经过这些策略的综合应用，我们的模型在A榜达到了0.5430的成绩，排名第五；在B榜取得了0.5159的成绩，名列第四。

### 方法的创新点
本项目的创新点如下：

1、构建多层级辅助监督分支。\
yolov9模型只在gelan模型中加入了一个辅助分支，没有对gelan模型的backbone中的路由信息进行进一步监督。
在此项目中，我们首先对每一个模态构建了2个辅助分支，分别对路由之前的特征和路由之后的特征进行监督，此外为了同时捕捉图像的低级和高级特征，我们对rgb和tir图像的高层和底层特征同时进行 \
辅助监督，并且交叉rgb和tir的不同层的特征进一步学习模态之间的交互信息。

2、使用灰度图弥补红外图的缺失。\
我们可以有多种方式将单模态数据集扩充为多模态数据集，包括直接将rgb图作为tir图、使用开源方法将rgb图转为tir图、使用全黑图作为tir图、
使用灰度图作为tir图，最终经过我们的测试，使用灰度图可以得到最优的结果，其他方式也有着次优的效果。

3、构建符合数据特点的数据增强策略。\
此赛题的数据存在以下特点：测试集和训练集差异较大、两张模态的图不对齐、场景多样性导致图片质量较差。\
我们针对以上数据特点，并针对不同模态的数据构建不同的数据增强方式，具体如下：\
（1）对rgb图像使用调亮和调暗增强。以模拟训练集中夜晚和白天场景。\
（2）对rgb和tir使用高斯噪声和椒盐噪声增强。以模拟训练集中阴雨、雨雾场景。\
（3）对tir使用模糊增强。以模拟红外相机拍摄导致的图片高度模糊现象。\
（4）对rgb和tir使用随机裁剪和旋转增强。以模拟车辆速度差异或相机拍摄时间差异导致的两种模态图片未对齐现象。

4、除以上共用的创新点之外，我们还进行了以下方法的探索，增加模型的多样性，最大化融合的收益。\
 (1) rgb 和 tir 的backbone特征进行对比学习，进行隐式的对齐。\
 (2) 对数据以随机的概率进行标签修改，增加噪声。
 (3) 对外部数据先进性目标检测范式的预训练，再训练比赛数据集。

5、多模型融合 \
我们为了提高模型融合带来的收益，从最大化结果差异的角度出发，训练出25个模型进行融合，最终得到相对于单模型在A榜提升1.8+的收益。

## 本项目仅供学术交流，禁止商用