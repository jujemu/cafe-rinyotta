from datetime import datetime
import os
import time
import warnings
warnings.filterwarnings(action='ignore')

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm
from scipy.spatial import distance
import pickle

import tag_infer


input_dir = './input/'

date_time = datetime.now().strftime("%Y-%m-%d_%H%M")
output_dir = os.path.join('./output/', date_time)
os.makedirs(output_dir, exist_ok=True)
model_url = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2" # feature map to embadding model
IMAGE_SHAPE = (224, 224)
metric = 'cosine' # how to calculate input vector with datasets vectors, it can be cosine, euclidean
emb_path = './embs.pkl'
with open(emb_path, 'rb') as f:
    emb = pickle.load(f)

arr_tag = ['vintage', 'modern', 'minimal', 'casual', 'whitewood']
tag2idx = {val: idx for idx, val in enumerate(arr_tag)}

layer = hub.KerasLayer(model_url)
model = tf.keras.Sequential([layer])


def extract(file):
    file = Image.open(file).convert('L').resize(IMAGE_SHAPE)
    file = np.stack((file,)*3, axis=-1)
    file = np.array(file)/255.0

    embedding = model.predict(file[np.newaxis, ...], verbose=0)
    vgg16_feature_np = np.array(embedding)
    flattended_feature = vgg16_feature_np.flatten()
    return flattended_feature


# arr_img_tag = tag_infer.classify_tag(input_dir=input_dir)
arr_img_tag = [('3.png', 'modern'), ('8.png', 'modern'), ('0.png', 'vintage'), ('6.png', 'vintage'), ('4.png', 'casual'), ('9.png', 'modern'), ('2.png', 'modern'), ('1.png', 'modern'), ('5.png', 'modern'), ('7.png', 'whitewood')]
print(f'the number of input images is {len(arr_img_tag)}')

# img : name of img
# tag : 'vintage', ... etcq
for img_name, tag in arr_img_tag:
    img_path = os.path.join(input_dir, img_name)
    nearest = None
    nearest_dst = np.Inf

    feature = extract(img_path)
    for i in range(5):
        for cafe in tqdm(emb[i]):
            dst = distance.cdist([feature], [cafe[1]], metric)[0]
            if nearest_dst > dst:
                nearest_dst = dst
                nearest = cafe[0]
        
        save_name = 'main_' + img_name if i == tag2idx[tag] else arr_tag[i]+'_'+img_name
        img_path = os.path.join('./rsc/', arr_tag[i], nearest)
        try:
            img = Image.open(img_path)
        except:
            print('Error in image path')
            print(img_path)
        img.save(os.path.join(output_dir, save_name))        
