"""Reuse local mini metadata and extract only its 404 LiDAR keyframes."""
import argparse
import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', default=r'D:\code\autodrive\stage02\PETR\v1.0-mini\v1.0-mini')
    parser.add_argument('--lidar-archive', default=r'D:\code\autodrive\nuscenes\nuscenes\samples\LIDAR_TOP.tar')
    parser.add_argument('--output', default=str(ROOT / 'data' / 'nuscenes'))
    args = parser.parse_args()
    metadata, output = Path(args.metadata), Path(args.output)
    tables = {name: json.loads((metadata / (name + '.json')).read_text(encoding='utf-8'))
              for name in ['sample', 'sample_data', 'calibrated_sensor', 'sensor']}
    calibrated = {x['token']: x for x in tables['calibrated_sensor']}
    sensors = {x['token']: x for x in tables['sensor']}
    records = [x for x in tables['sample_data'] if x['is_key_frame'] and
               sensors[calibrated[x['calibrated_sensor_token']]['sensor_token']]['channel'] == 'LIDAR_TOP']
    assert len(records) == len(tables['sample']) == 404
    wanted = {Path(x['filename']).name for x in records}
    destination = output / 'samples' / 'LIDAR_TOP'
    destination.mkdir(parents=True, exist_ok=True)
    meta_out = output / 'v1.0-mini'
    shutil.copytree(metadata, meta_out, dirs_exist_ok=True)
    maps_source = metadata.parent / 'maps'
    if maps_source.is_dir():
        shutil.copytree(maps_source, output / 'maps', dirs_exist_ok=True)
    found = set()
    with tarfile.open(args.lidar_archive) as archive:
        for member in archive:
            name = Path(member.name).name
            if member.isfile() and name in wanted:
                target = destination / name
                if not target.is_file() or target.stat().st_size != member.size:
                    with archive.extractfile(member) as source, target.open('wb') as dest:
                        shutil.copyfileobj(source, dest)
                assert target.stat().st_size % 20 == 0 and target.stat().st_size > 0
                found.add(name)
                if len(found) == len(wanted):
                    break
    assert found == wanted, f'Missing {len(wanted - found)} point files'
    audit = dict(metadata_source=str(metadata), lidar_archive=args.lidar_archive,
                 dataset_version='v1.0-mini', keyframes=len(found), sweeps_extracted=0,
                 raw_storage_dimensions=5, model_input_dimensions=4)
    (output / 'local_data_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print('LOCAL_DATA_OK keyframes=404 sweeps_extracted=0', flush=True)


if __name__ == '__main__':
    main()
