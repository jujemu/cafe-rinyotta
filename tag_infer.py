import os
from glob import glob

import click
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch_xla.core.xla_model as xm
from tqdm import tqdm

from recsys_cafe.final_model.tag.tag_mobilenet import SuperLightMobileNet

# parameter
num_cls = 5
arr_cls = ['casual', 'minimal', 'modern', 'vintage', 'whitewood']
model_path = './model/model_weights.pth'
save_dir = './output/'
os.makedirs(save_dir, exist_ok=True)

@click.command()
@click.option('--network', 'network', default=model_path)
@click.option('--sample', 'sample_path', default=None)
@click.option('--dir', 'input_dir', default='./input/')
@click.option('--device', 'device', default=None)
def main(
    network=model_path,
    sample_path=None,
    input_dir=None,
    device=None,
):
    classify_tag(
        network,
        sample_path,
        input_dir,
        device
    )


def classify_tag(
    network=model_path,
    sample_path=None,
    input_dir=None,
    device=None,
):
    if not device:
        device = xm.xla_device()
    print(f'device is {device}')
    # use model you want
    # this script uses resnet50 model pretrained
    # model = SuperLightMobileNet(num_classes).to(device)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50')
    model.fc = nn.Linear(2048, num_cls)
    model = model.to(device)
    model.load_state_dict(torch.load(network))    
    print(f'model is prepared with model_weight you give')

    # transforms input images
    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    if sample_path:
        print("you put in one image, Now begin to classifying image's tag")
        sample_tensor = transforms(Image.open(sample_path)).to(device)
        sample_tensor = torch.unsqueeze(sample_tensor, 0)
        output = model(sample_tensor)
        output = arr_cls[torch.argmax(output).cpu().numpy()]
        print(f'Sample image is "{output}" tag class')
        return

    assert os.path.isdir(input_dir), 'put in directory of input images'
    result = []
    for path in tqdm(glob(input_dir+'*'), total=len(os.listdir(input_dir))):
        sample_tensor = transforms(Image.open(path).convert("RGB")).to(device)
        sample_tensor = torch.unsqueeze(sample_tensor, 0)
        output = model(sample_tensor)
        output = arr_cls[torch.argmax(output).cpu().numpy()]

        img_name = path.split('/')[-1]
        result.append(tuple((img_name, output)))
    print('Display output of classifying tag')
    print('image_name, classes of tag')
    print(*result, sep=', ')
    if len(result) > 5:
        print('to prevent from being verbose, display only 5 imgs')
    return result


if __name__ == '__main__':
    main()
