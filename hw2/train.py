from model      import cnn_model
from config      import cfg
from datasets import make_train_loader
import datetime
import torch.nn as nn  
import torchvision.models as models
import torch, os
import numpy as np

model = models.resnet50()
fc_features = model.fc.in_features
model.fc = nn.Linear(fc_features, 3)


valid_size  = cfg.DATA.VALIDATION_SIZE
epochs      = cfg.MODEL.EPOCH
lr          = cfg.MODEL.LR
weight_path = cfg.MODEL.OUTPUT_PATH
use_cuda    = cfg.DEVICE.CUDA
gpu_id      = cfg.DEVICE.GPU

if use_cuda:
    torch.cuda.set_device(gpu_id)
    model = model.cuda()

train_loader, valid_loader = make_train_loader(cfg)

optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=0.0001)

for epoch in range(1, epochs+1):
    model.train()
    train_loss = 0.
    valid_loss = 0.

    for data, target in train_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    model.eval()
    for data, target in valid_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
            
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        valid_loss += loss.item() * data.size(0)

    train_loss /= int(np.floor(len(train_loader.dataset) * (1 - valid_size)))
    valid_loss /= int(np.floor(len(valid_loader.dataset) * valid_size))
    print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, train_loss, valid_loss))
    output_dir = "\\".join(weight_path.split("\\")[:-1])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), weight_path.replace(".pth",str(epoch)+".pth"))
    print(datetime.datetime.now())

output_dir = "\\".join(weight_path.split("\\")[:-1])
if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
torch.save(model.state_dict(), weight_path)
