"""Plot actual loss history and evaluate center-camera predictions on held-out rows."""
import csv,json,sys
from pathlib import Path
import cv2,numpy as np,torch,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,'upstream/src')
from model import NetworkNvidia
from train import DrivingData
from torch.utils.data import DataLoader
p=Path('outputs/run30'); h=json.loads((p/'history.json').read_text()); fig,ax=plt.subplots(figsize=(7,4)); ax.plot([x['epoch'] for x in h],[x['train_mse'] for x in h],'-o',label='Training: three cameras'); ax.plot([x['epoch'] for x in h],[x['val_mse'] for x in h],'-o',label='Validation: center camera'); ax.set(xlabel='Epoch',ylabel='Mean squared error'); ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(p/'loss.png',dpi=170); plt.close(fig)
rows=list(csv.reader(open('dataset/driving_log.csv',encoding='utf-8-sig',newline=''))); order=np.random.RandomState(42).permutation(len(rows)); val=[rows[i] for i in order[int(.8*len(rows)):]]
model=NetworkNvidia().cuda(); ck=torch.load(p/'model-best.h5',map_location='cuda'); model.load_state_dict(ck['state_dict']); model.eval(); pred=[]; truth=[]
with torch.no_grad():
 for x,y in DataLoader(DrivingData(val,Path('dataset'),False),batch_size=64,num_workers=0): pred.extend(model(x.cuda()).cpu().flatten().tolist()); truth.extend(y.flatten().tolist())
pred=np.asarray(pred); truth=np.asarray(truth); result={'checkpoint_epoch':ck['epoch'],'samples':len(pred),'mse':float(np.mean((pred-truth)**2)),'mae':float(np.mean(np.abs(pred-truth))),'rmse':float(np.sqrt(np.mean((pred-truth)**2))),'prediction_mean':float(pred.mean()),'label_mean':float(truth.mean())}; (p/'evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result)
fig,ax=plt.subplots(figsize=(6,5)); ax.hist2d(truth,pred,bins=80,cmap='Greys',norm=matplotlib.colors.LogNorm()); ax.plot([-1,1],[-1,1],'k--'); ax.set(xlabel='Recorded steering',ylabel='Predicted steering',xlim=(-1,1),ylim=(-1,1)); fig.tight_layout(); fig.savefig(p/'predictions.png',dpi=170)
