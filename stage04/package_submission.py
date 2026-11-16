from pathlib import Path
import shutil, zipfile
root=Path('.'); dest=root/'output/submission/自动驾驶技术第四次实验_杜承敖_2024302112019'
if dest.exists(): raise FileExistsError(dest)
(dest/'代码').mkdir(parents=True); (dest/'结果').mkdir(); (dest/'报告').mkdir()
for name in ['README.md','train.py','evaluate.py','check_pipeline.py','check_socket.py']:
 shutil.copy2(root/name,dest/'代码'/name)
shutil.copytree(root/'upstream/src',dest/'代码/upstream/src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ['LICENSE','README.md']:
 if (root/'upstream'/name).exists(): shutil.copy2(root/'upstream'/name,dest/'代码/upstream'/name)
for name in ['history.json','config.json','evaluation.json','loss.png','predictions.png','model-best.h5']:
 shutil.copy2(root/'outputs/run30'/name,dest/'结果'/name)
for name in ['drive.log','synthetic_telemetry.json','simulator_status.txt']:
 shutil.copy2(root/'outputs'/name,dest/'结果'/name)
shutil.copy2(root/'outputs_train.log',dest/'结果/训练1至5轮.log'); shutil.copy2(root/'outputs_train30.log',dest/'结果/训练6至30轮.log')
shutil.copy2(root/'output/pdf/自动驾驶技术第四次实验报告_杜承敖_2024302112019.pdf',dest/'报告')
archive=dest.with_suffix('.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in dest.rglob('*'):
  if p.is_file(): z.write(p,p.relative_to(dest.parent))
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
print(archive,archive.stat().st_size)
