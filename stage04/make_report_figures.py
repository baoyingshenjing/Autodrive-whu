from pathlib import Path
import csv
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
out=Path('output/pdf/figures');out.mkdir(parents=True,exist_ok=True)
with open('dataset/driving_log.csv',encoding='utf-8-sig',newline='') as f: rows=list(csv.reader(f))
# Pick a recorded frame with a moderate turn and verify all three views exist.
row=next(r for r in rows if .15 < abs(float(r[3])) < .25 and all((Path('dataset/IMG')/x.strip().replace('\\','/').split('/')[-1]).exists() for x in r[:3]))
imgs=[cv2.cvtColor(cv2.imread(str(Path('dataset/IMG')/x.strip().replace('\\','/').split('/')[-1])),cv2.COLOR_BGR2RGB) for x in row[:3]]
fig,axs=plt.subplots(1,3,figsize=(10,2));
for ax,img,title in zip(axs,imgs,['Center','Left','Right']): ax.imshow(img); ax.set_title(title,fontsize=13); ax.axis('off')
fig.tight_layout();fig.savefig(out/'camera_views.png',dpi=180,bbox_inches='tight');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(8,2.2)); axs[0].imshow(imgs[0]);axs[0].axhline(65,color='yellow',linewidth=1.2);axs[0].axhline(135,color='yellow',linewidth=1.2);axs[0].set_title('Original (160 x 320)',fontsize=13);axs[1].imshow(imgs[0][65:-25]);axs[1].set_title('Crop (70 x 320)',fontsize=13)
for ax in axs: ax.axis('off')
fig.tight_layout();fig.savefig(out/'crop.png',dpi=180,bbox_inches='tight');plt.close(fig)
angles=np.array([float(r[3]) for r in rows]);fig,ax=plt.subplots(figsize=(7,2.25));ax.tick_params(labelsize=11); ax.hist(angles,bins=80,color='#4d5964',edgecolor='none');ax.set(xlabel='Recorded center steering',ylabel='Frame count');ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(out/'steering_hist.png',dpi=180,bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(10,1.6));ax.set_xlim(0,10);ax.set_ylim(0,1);ax.axis('off')
labels=['Input\n3 x 70 x 320','Conv1-3\n48 x 6 x 37','Conv4-5\n64 x 2 x 33','Flatten\n4224','FC layers\n100 / 50 / 10','Output\n1 steering']
for i,label in enumerate(labels):
 x=.08+i*1.65; ax.add_patch(FancyBboxPatch((x,.25),1.4,.5,boxstyle='round,pad=0.03',facecolor='#edf0f2',edgecolor='#333333'));ax.text(x+.7,.5,label,ha='center',va='center',fontsize=12)
 if i<5:ax.annotate('',xy=(x+1.63,.5),xytext=(x+1.42,.5),arrowprops=dict(arrowstyle='->',color='#333333'))
fig.tight_layout();fig.savefig(out/'architecture.png',dpi=180,bbox_inches='tight');plt.close(fig)
print('sample steering:',row[3]);print('images:',len(angles),'zero fraction:',float(np.mean(angles==0)))
