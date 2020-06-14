from model      import cnn_model
from config     import cfg
from datasets   import make_nolabeltest_loader
import torch.nn as nn  
import torchvision.models as models

import torch, os
import numpy as np
import mongo_data.rotation 

model = models.resnet50()
fc_features = model.fc.in_features  
model.fc = nn.Linear(fc_features, 3)

weight_path = cfg.MODEL.OUTPUT_PATH
use_cuda    = cfg.DEVICE.CUDA
gpu_id      = cfg.DEVICE.GPU

weight = torch.load(weight_path)
model.load_state_dict(weight)

if use_cuda:
    torch.cuda.set_device(gpu_id)
    model.cuda()

test_loader = make_nolabeltest_loader(cfg)

model.eval()
result_label=[]
with torch.no_grad():
    for data, target in test_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        output = model(data)
        print(output)
        ndarray = output.max(1)[1].cpu().numpy()
        print(ndarray)
        k = ndarray.tolist()
        print(k)
        print('------')
        if (k[0] == 0):
            result_label.append('A')
        if (k[0] == 1):
            result_label.append('B')
        if (k[0] == 2):
            result_label.append('C')
       
print(result_label)
import csv
with open("result_label.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(result_label)
