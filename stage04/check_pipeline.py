from pathlib import Path
import csv, sys
import cv2, numpy as np, torch
sys.path.insert(0,'upstream/src')
from train import DrivingData
from model import NetworkNvidia
row=next(csv.reader(open('dataset/driving_log.csv',encoding='utf-8-sig')))
x,y=DrivingData([row],Path('dataset'),False)[0]
image=cv2.imread(str(Path('dataset/IMG')/row[0].replace('\\','/').split('/')[-1]))
simulator_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
expected=((simulator_rgb[65:-25,:,::-1].astype(np.float32)/127.5)-1).transpose(2,0,1)
assert x.shape==(3,70,320) and np.allclose(x.numpy(),expected)
assert torch.isfinite(NetworkNvidia()(x.unsqueeze(0))).all()
print('PASS: BGR training and RGB simulator preprocessing agree; shape and model forward OK')
