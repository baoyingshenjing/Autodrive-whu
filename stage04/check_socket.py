import base64,csv,json,time
from pathlib import Path
import socketio
row=next(csv.reader(open('dataset/driving_log.csv',encoding='utf-8-sig')))
img=Path('dataset/IMG')/row[0].replace('\\','/').split('/')[-1]
sio=socketio.Client(); controls=[]
@sio.on('steer')
def steer(data):
 controls.append(data); print('control:',data,flush=True)
sio.connect('http://127.0.0.1:4567',transports=['polling'])
sio.emit('telemetry',{'steering_angle':'0','throttle':'0','speed':'5','image':base64.b64encode(img.read_bytes()).decode()})
for _ in range(50):
 if len(controls) >= 2: break
 time.sleep(.1)
sio.disconnect()
if len(controls) < 2: raise RuntimeError('no steer response')
Path('outputs/synthetic_telemetry.json').write_text(json.dumps({'test':'synthetic client, not simulator','response':controls[-1]},indent=2))

