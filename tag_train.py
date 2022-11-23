from collections import Counter
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torch_xla.core.xla_model as xm
from tqdm import tqdm

from recsys_cafe.final_model.tag.tag_mobilenet import SuperLightMobileNet

os.makedirs('./model', exist_ok=True)

# parameter
device = xm.xla_device()
num_classes = 5
batch_size = 128
lr = 1e-6
total_epoch = 100
big_cls = 'total'

# label
# dataset.class_to_idx
# {'casual': 0, 'minimal': 1, 'modern': 2, 'vintage': 3, 'whitewood': 4}
ds_path ='./rsc/'
ds = ImageFolder(
    ds_path,
    T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor()
    ]))

ds_len = len(ds)
indices = np.random.choice(np.arange(ds_len), size=(ds_len, ), replace=False)
val_size = ds_len // 5
train_idx = indices[:val_size]
val_idx = indices[val_size:]

train_loader = DataLoader(ds, batch_size, sampler=train_idx)
val_loader = DataLoader(ds, batch_size, sampler=val_idx)
print(f'train dataloader is prepared; {len(train_loader)}, {train_idx[:3]} ...')
print(f'validation dataloader is prepared; {len(val_loader)}, {val_idx[:3]} ...')


if __name__ == "__main__":
    weights = [1/i for i in list(Counter(ds.targets).values())] #[ 1 / number of instances for each class]
    class_weights = torch.FloatTensor(weights)

    # model = SuperLightMobileNet(num_classes).to(device)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    model.fc = nn.Linear(2048, len(ds.classes))
    model = model.to(device)

    CEloss = nn.CrossEntropyLoss(weight=class_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_iteration_per_epoch = int(np.ceil(len(train_idx)/batch_size))

    for epoch in range(1, total_epoch + 1):
        model.train()
        with tqdm(train_loader, total=len(train_loader), desc='Process is in ' + str(epoch).zfill(2)) as iterator:
            for itereation, (input, target) in enumerate(iterator):
                images = input.to(device)
                labels = target.to(device)

                # Forward pass
                outputs = model(images)
                loss = CEloss(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                xm.mark_step()

                # display loss in sys.out
                log = f'loss : {loss}'
                iterator.set_postfix_str(log)
        save_weight = str(epoch) + '_model_weights.pth'
        torch.save(model.state_dict(), os.path.join('./model', save_weight))
                  