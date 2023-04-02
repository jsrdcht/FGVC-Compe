from typing import TYPE_CHECKING, Union

import os
import glob
import csv
from tqdm import tqdm

from PIL import Image
import numpy as np

import torch

from mmcls.apis import inference_model, init_model
from mmcls.utils import register_all_modules

if TYPE_CHECKING:
    from mmengine.model import BaseModel

config_path = '/home/ct/code/fgvc/mmclassification/configs/fgvc/greedyhash_vit-base-p16_pt-64xb64_iBioHash1k-224.py'
checkpoint_path = '/home/ct/code/fgvc/mmclassification/results/test/epoch_30.pth' # 也可以设置为一个本地的路径
query_path = '/home/ct/code/fgvc/iBioHash/Query'
gallery_path = '/home/ct/code/fgvc/iBioHash/Gallery'

# 注册
register_all_modules()                 # 将所有模块注册在默认 mmcls 域中
# 构建模型
model = init_model(config_path, checkpoint_path, device="cuda:0") # `device` 可以为 'cuda:0'

def inference_model(model: 'BaseModel', img: Union[str, np.ndarray]):
    """Inference image(s) with the classifier.

    Args:
        model (BaseClassifier): The loaded classifier.
        img (str/ndarray): The image filename or loaded image.

    Returns:
        result (dict): The classification results that contains
            `class_name`, `pred_label` and `pred_score`.
    """
    from mmengine.dataset import Compose, default_collate
    from mmengine.registry import DefaultScope

    import mmcls.datasets  # noqa: F401

    cfg = model.cfg
    # build the data pipeline
    test_pipeline_cfg = [
    dict(type='LoadImageFromFile'),     # 读取图像
    dict(type='ResizeEdge', scale=256, edge='short'),  # 短边对其256进行放缩
    dict(type='CenterCrop', crop_size=224),     # 中心裁剪
    dict(type='PackClsInputs'),                 # 准备图像以及标签
]

    if isinstance(img, str):
        if test_pipeline_cfg[0]['type'] != 'LoadImageFromFile':
            test_pipeline_cfg.insert(0, dict(type='LoadImageFromFile'))
        data = dict(img_path=img)
    else:
        if test_pipeline_cfg[0]['type'] == 'LoadImageFromFile':
            test_pipeline_cfg.pop(0)
        data = dict(img=img)
    with DefaultScope.overwrite_default_scope('mmcls'):
        test_pipeline = Compose(test_pipeline_cfg)
    data = test_pipeline(data)
    data = default_collate([data])

    # forward the model
    with torch.no_grad():
        hash_code = model.val_step(data)[0].hash_code

        result = hash_code
       
    return result

# 执行推理，为query和gallery生成hashcode。格式为submit_query.csv([image_id, hashcode])和submit_gallery.csv([image_id, hashcode])
def gen_hashcode(model, img_path, save_file_name):
    with open(save_file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['image_id', 'hashcode'])

        # 遍历img_path下的所有图片
        img_files = list(glob.glob(os.path.join(img_path, '*.jpg')))
        for img_file in tqdm(img_files, desc="Processing images"):
            img_id = os.path.basename(img_file).split('.')[0]
            hashcode = inference_model(model=model, img=img_file)

            # 将图片ID和hashcode写入CSV文件
            csv_writer.writerow([img_id, hashcode])
def gen_hashcode(model, img_path, save_file_name):
    with open(save_file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['image_id', 'hashcode'])

        # 遍历img_path下的所有图片
        img_files = list(glob.glob(os.path.join(img_path, '*.jpg')))
        for img_file in tqdm(img_files, desc="Processing images"):
            img_id = os.path.basename(img_file)
            hashcode = inference_model(model=model, img=img_file)

            # 将图片ID和hashcode写入CSV文件
            # 将 hashcode 转换为字符串，并使用单引号包围
            csv_writer.writerow([img_id, f"'{hashcode}'"])


# 为query和gallery路径下的所有图片生成hashcode，并将结果保存为CSV文件
gen_hashcode(model, query_path, 'submit_query.csv')
gen_hashcode(model, gallery_path, 'submit_gallery.csv')
    
    