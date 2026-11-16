"""Reproducible NVIDIA behavioral cloning on the supplied three-camera dataset."""
import argparse, csv, json, random, sys, time
from pathlib import Path
import cv2, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, str(Path(__file__).parent / 'upstream' / 'src'))
from model import NetworkNvidia

class DrivingData(Dataset):
    def __init__(self, rows, root, train): self.rows, self.root, self.train = rows, root, train
    def __len__(self): return len(self.rows) * (3 if self.train else 1)
    def __getitem__(self, idx):
        row = self.rows[idx // 3] if self.train else self.rows[idx]
        camera = idx % 3 if self.train else 0
        path = self.root / 'IMG' / row[camera].strip().replace('\\', '/').split('/')[-1]
        img = cv2.imread(str(path))
        if img is None: raise FileNotFoundError(path)
        angle = float(row[3]) + (0, .4, -.4)[camera]
        img = img[65:-25]
        if self.train and random.random() < .5:
            img = cv2.flip(img, 1); angle = -angle
        img = np.ascontiguousarray((img.astype(np.float32) / 127.5 - 1).transpose(2,0,1))
        return torch.from_numpy(img), torch.tensor([angle], dtype=torch.float32)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=5); ap.add_argument('--batch-size',type=int,default=64); ap.add_argument('--workers',type=int,default=4); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--resume',type=Path); ap.add_argument('--data',type=Path,default=Path('dataset')); ap.add_argument('--output',type=Path,default=Path('outputs/run')); a=ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.set_num_threads(4); cv2.setNumThreads(0)
    rows=list(csv.reader((a.data/'driving_log.csv').open(newline='',encoding='utf-8-sig')))
    order=np.random.RandomState(a.seed).permutation(len(rows)); n=int(.8*len(rows)); train=[rows[i] for i in order[:n]]; val=[rows[i] for i in order[n:]]
    a.output.mkdir(parents=True,exist_ok=False)
    info={'seed':a.seed,'rows':len(rows),'train_rows':len(train),'val_rows':len(val),'train_views':len(train)*3,'val_views':len(val),'batch_size':a.batch_size,'epochs':a.epochs,'lr':1e-4,'weight_decay':1e-5,'validation':'center only, no augmentation; random split by record','device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu','torch':torch.__version__}
    (a.output/'config.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(info),flush=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=NetworkNvidia().to(device); opt=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5); sched=torch.optim.lr_scheduler.MultiStepLR(opt,[30,50],gamma=.1); lossfn=torch.nn.MSELoss(reduction='sum')
    trainloader=DataLoader(DrivingData(train,a.data,True),batch_size=a.batch_size,shuffle=True,num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0)
    valloader=DataLoader(DrivingData(val,a.data,False),batch_size=a.batch_size,shuffle=False,num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0)
    history=[]; best=float('inf'); start=0
    if a.resume:
        ck=torch.load(a.resume,map_location=device); model.load_state_dict(ck['state_dict']); opt.load_state_dict(ck['optimizer']); sched.load_state_dict(ck['scheduler']); start=ck['epoch']
        history=json.loads((a.resume.parent/'history.json').read_text()); best=min(x['val_mse'] for x in history)
        import shutil; shutil.copyfile(a.resume.parent/'model-best.h5', a.output/'model-best.h5')
    for epoch in range(start+1,a.epochs+1):
        t=time.time(); model.train(); total=0.; count=0
        for x,y in trainloader:
            x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True); opt.zero_grad(set_to_none=True); loss=lossfn(model(x),y); (loss/len(x)).backward(); opt.step(); total+=loss.item(); count+=len(x)
        model.eval(); vt=0.; vc=0
        with torch.no_grad():
            for x,y in valloader:
                x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True); vt+=lossfn(model(x),y).item(); vc+=len(x)
        record={'epoch':epoch,'train_mse':total/count,'val_mse':vt/vc,'lr':opt.param_groups[0]['lr'],'seconds':time.time()-t}; history.append(record); sched.step()
        state={'epoch':epoch,'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'config':info}
        torch.save(state,a.output/f'model-{epoch}.h5')
        if vt/vc < best: best=vt/vc; torch.save(state,a.output/'model-best.h5')
        (a.output/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8'); print(json.dumps(record),flush=True)
if __name__=='__main__': main()

