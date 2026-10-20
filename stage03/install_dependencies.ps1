$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:TEMP = Join-Path $PSScriptRoot 'tmp'
$env:TMP = $env:TEMP
$env:PYTHONUTF8 = '1'
$taskPython = Join-Path $PSScriptRoot '.env\python.exe'
& $taskPython -m pip install numpy==1.24.4 scipy==1.13.1 matplotlib==3.5.3 opencv-python==4.8.1.78 mmengine==0.10.7 mmdet==3.2.0 nuscenes-devkit==1.1.11 numba==0.60.0 plyfile==1.1.3 scikit-image==0.24.0 tensorboard==2.14.0 trimesh==4.4.9 yapf==0.40.1 --cache-dir downloads --progress-bar off --index-url https://mirrors.aliyun.com/pypi/simple/
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $taskPython -m pip install mmcv==2.1.0 --only-binary=mmcv -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html --cache-dir downloads --progress-bar off --index-url https://mirrors.aliyun.com/pypi/simple/
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $taskPython -m pip install lyft-dataset-sdk==0.0.8 --no-deps --cache-dir downloads --progress-bar off --index-url https://mirrors.aliyun.com/pypi/simple/
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if (Test-Path -LiteralPath 'downloads\open3d-0.18.0-cp39-cp39-win_amd64.whl') {
    $taskOpen3d = 'downloads\open3d-0.18.0-cp39-cp39-win_amd64.whl'
} else {
    $taskOpen3d = 'open3d==0.18.0'
}
& $taskPython -m pip install $taskOpen3d --cache-dir downloads --progress-bar off --index-url https://mirrors.aliyun.com/pypi/simple/
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $taskPython -m pip install .\third_party\mmdetection3d-1.4.0 --no-deps --no-build-isolation --cache-dir downloads --progress-bar off --index-url https://mirrors.aliyun.com/pypi/simple/
exit $LASTEXITCODE

