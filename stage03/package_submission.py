"""Package only the verified report, reproducible sources and required evidence."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUB = ROOT / 'output' / 'submission'
NAME = '自动驾驶技术第三次实验_杜承敖_2024302112019'


def main():
    validation = json.loads((ROOT / 'outputs' / 'final_validation.json').read_text(encoding='utf-8'))
    destination = SUB / NAME
    destination.mkdir(parents=True, exist_ok=True)
    def copy(path):
        relative = path.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for path in ROOT.glob('*.py'):
        copy(path)
    for name in ['README.md', 'requirements-course.txt', 'install_dependencies.ps1', 'Pointpillar指导文档.pdf']:
        copy(ROOT / name)
    for folder in ['checkpoints', 'data_infos', 'output/pdf', 'third_party/mmdetection3d-1.4.0']:
        for path in (ROOT / folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and '.pyc' != path.suffix:
                if 'pointpillars-latex' in path.parts and path.suffix in {'.aux','.log','.out','.toc','.pdf','.fls','.fdb_latexmk'}:
                    continue
                copy(path)
    for path in (ROOT / 'outputs').rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        if path.suffix == '.pth' and path.name not in ['epoch_1.pth', 'epoch_5.pth']:
            continue
        copy(path)
    for name in ['local_data_audit.json', 'source_infos_audit.json']:
        copy(ROOT / 'data' / 'nuscenes' / name)
    files = []
    for path in sorted(destination.rglob('*')):
        if path.is_file() and path.name != '提交清单.json':
            files.append(dict(path=path.relative_to(destination).as_posix(),
                              bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    required = ['outputs/run5_compact/epoch_5.pth', 'outputs/comparison.json', 'data_infos/manifest.json']
    assert all((destination / name).is_file() for name in required)
    (destination / '提交清单.json').write_text(json.dumps(dict(files=files, final_checkpoint_sha256=validation['final_checkpoint_sha256']), ensure_ascii=False, indent=2), encoding='utf-8')
    archive = SUB / (NAME + '.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as stream:
        for path in sorted(destination.rglob('*')):
            if path.is_file():
                stream.write(path, path.relative_to(SUB).as_posix())
    with zipfile.ZipFile(archive) as stream:
        assert stream.testzip() is None
    print('SUBMISSION_OK', archive, archive.stat().st_size)


if __name__ == '__main__':
    main()
