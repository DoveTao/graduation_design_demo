# MAINT1 Conda Environment Inventory

## Host Diagnostics

### `which conda`

```text
/home/dovetao/miniconda3/condabin/conda
```

### `conda info`

```text

     active environment : None
            shell level : 0
       user config file : /home/dovetao/.condarc
 populated config files : /home/dovetao/miniconda3/.condarc
                          /home/dovetao/miniconda3/condarc.d/anaconda-auth.yml
                          /home/dovetao/.condarc
          conda version : 25.11.1
    conda-build version : not installed
         python version : 3.13.11.final.0
                 solver : libmamba (default)
       virtual packages : __archspec=1=skylake
                          __conda=25.11.1=0
                          __cuda=13.0=0
                          __glibc=2.35=0
                          __linux=6.8.0=0
                          __unix=0=0
       base environment : /home/dovetao/miniconda3  (writable)
      conda av data dir : /home/dovetao/miniconda3/etc/conda
  conda av metadata url : None
           channel URLs : https://repo.anaconda.com/pkgs/main/linux-64
                          https://repo.anaconda.com/pkgs/main/noarch
                          https://repo.anaconda.com/pkgs/r/linux-64
                          https://repo.anaconda.com/pkgs/r/noarch
          package cache : /home/dovetao/miniconda3/pkgs
                          /home/dovetao/.conda/pkgs
       envs directories : /home/dovetao/miniconda3/envs
                          /home/dovetao/.conda/envs
               platform : linux-64
             user-agent : conda/25.11.1 requests/2.32.5 CPython/3.13.11 Linux/6.8.0-111-generic ubuntu/22.04.5 glibc/2.35 solver/libmamba conda-libmamba-solver/25.11.0 libmambapy/2.3.2 aau/0.7.5 c/. s/. e/.
                UID:GID : 1000:1000
             netrc file : None
           offline mode : False
```

### `nvidia-smi`

```text
Wed May 13 02:38:36 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.142                Driver Version: 580.142        CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3060 ...    Off |   00000000:01:00.0  On |                  N/A |
| N/A   58C    P3             26W /  100W |    1224MiB /   6144MiB |      9%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            5422      G   /usr/lib/xorg/Xorg                      478MiB |
|    0   N/A  N/A            5601      G   /usr/bin/gnome-shell                     84MiB |
|    0   N/A  N/A            7367      G   clash-verge                               2MiB |
|    0   N/A  N/A            9529      G   .../8274/usr/lib/firefox/firefox         23MiB |
|    0   N/A  N/A           10133      G   ...rack-uuid=3190708988185955192        240MiB |
|    0   N/A  N/A           10711      G   /usr/share/code/code                    253MiB |
|    0   N/A  N/A           13239      G   ...ns-seed-version --log-level=2          6MiB |
+-----------------------------------------------------------------------------------------+
```

### `nvcc --version`

```text
/bin/bash: 行 1: nvcc: 未找到命令
```

### `gcc --version`

```text
gcc (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0
Copyright (C) 2021 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

### `g++ --version`

```text
g++ (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0
Copyright (C) 2021 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

### `conda env list`

```text

# conda environments:
#
# * -> active
# + -> frozen
base                     /home/dovetao/miniconda3
base360dvo               /home/dovetao/miniconda3/envs/base360dvo
base360dvo_rebuild       /home/dovetao/miniconda3/envs/base360dvo_rebuild
pytorch                  /home/dovetao/miniconda3/envs/pytorch
```

## Environment: `pytorch`

```text
python: /home/dovetao/miniconda3/envs/pytorch/bin/python
version: 3.12.12 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 20:16:04) [GCC 11.2.0]
torch: import ok 2.9.1+cu128
cv2: import ok 4.13.0
numpy: import ok 2.4.4
scipy: import ok 1.17.1
evo: import ok v1.36.4
viser: import ok 1.0.27
torch cuda: 12.8
cuda available: True
device count: 1
device 0: NVIDIA GeForce RTX 3060 Laptop GPU
torch cxx11 abi: True
cuda_ba: import failed: ModuleNotFoundError("No module named 'cuda_ba'")
JSON_RESULT={"python": "/home/dovetao/miniconda3/envs/pytorch/bin/python", "version": "3.12.12 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 20:16:04) [GCC 11.2.0]", "imports": {"torch": {"ok": true, "version": "2.9.1+cu128"}, "cv2": {"ok": true, "version": "4.13.0"}, "numpy": {"ok": true, "version": "2.4.4"}, "scipy": {"ok": true, "version": "1.17.1"}, "evo": {"ok": true, "version": "v1.36.4"}, "viser": {"ok": true, "version": "1.0.27"}}, "torch": {"cuda": "12.8", "cuda_available": true, "device_count": 1, "device_0": "NVIDIA GeForce RTX 3060 Laptop GPU", "cxx11_abi": true}, "cuda_ba": {"ok": false, "error": "ModuleNotFoundError(\"No module named 'cuda_ba'\")"}}
```

## Environment: `base360dvo`

```text
python: /home/dovetao/miniconda3/envs/base360dvo/bin/python
version: 3.11.15 (main, Mar 11 2026, 17:20:07) [GCC 14.3.0]
torch: import failed: ImportError('/home/dovetao/miniconda3/envs/base360dvo/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent')
cv2: import failed: ModuleNotFoundError("No module named 'cv2'")
numpy: import ok 2.4.4
scipy: import failed: ModuleNotFoundError("No module named 'scipy'")
evo: import failed: ModuleNotFoundError("No module named 'evo'")
viser: import failed: ModuleNotFoundError("No module named 'viser'")
torch details failed: ImportError('/home/dovetao/miniconda3/envs/base360dvo/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent')
cuda_ba: import failed: ModuleNotFoundError("No module named 'cuda_ba'")
JSON_RESULT={"python": "/home/dovetao/miniconda3/envs/base360dvo/bin/python", "version": "3.11.15 (main, Mar 11 2026, 17:20:07) [GCC 14.3.0]", "imports": {"torch": {"ok": false, "error": "ImportError('/home/dovetao/miniconda3/envs/base360dvo/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent')"}, "cv2": {"ok": false, "error": "ModuleNotFoundError(\"No module named 'cv2'\")"}, "numpy": {"ok": true, "version": "2.4.4"}, "scipy": {"ok": false, "error": "ModuleNotFoundError(\"No module named 'scipy'\")"}, "evo": {"ok": false, "error": "ModuleNotFoundError(\"No module named 'evo'\")"}, "viser": {"ok": false, "error": "ModuleNotFoundError(\"No module named 'viser'\")"}}, "torch": {}, "cuda_ba": {"ok": false, "error": "ModuleNotFoundError(\"No module named 'cuda_ba'\")"}, "torch_error": "ImportError('/home/dovetao/miniconda3/envs/base360dvo/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent')"}
```

## Environment: `base360dvo_rebuild`

```text
python: /home/dovetao/miniconda3/envs/base360dvo_rebuild/bin/python
version: 3.11.15 (main, Mar 11 2026, 17:20:07) [GCC 14.3.0]
torch: import failed: ImportError('/home/dovetao/miniconda3/envs/base360dvo_rebuild/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent')
cv2: import ok 4.13.0
numpy: import ok 1.26.4
scipy: import ok 1.17.1
evo: import ok v1.36.4
viser: import ok 1.0.27
torch details failed: ImportError('/home/dovetao/miniconda3/envs/base360dvo_rebuild/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent')
cuda_ba: import failed: ImportError('libc10.so: cannot open shared object file: No such file or directory')
JSON_RESULT={"python": "/home/dovetao/miniconda3/envs/base360dvo_rebuild/bin/python", "version": "3.11.15 (main, Mar 11 2026, 17:20:07) [GCC 14.3.0]", "imports": {"torch": {"ok": false, "error": "ImportError('/home/dovetao/miniconda3/envs/base360dvo_rebuild/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent')"}, "cv2": {"ok": true, "version": "4.13.0"}, "numpy": {"ok": true, "version": "1.26.4"}, "scipy": {"ok": true, "version": "1.17.1"}, "evo": {"ok": true, "version": "v1.36.4"}, "viser": {"ok": true, "version": "1.0.27"}}, "torch": {}, "cuda_ba": {"ok": false, "error": "ImportError('libc10.so: cannot open shared object file: No such file or directory')"}, "torch_error": "ImportError('/home/dovetao/miniconda3/envs/base360dvo_rebuild/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent')"}
```

## Environment: `base`

```text
python: /home/dovetao/miniconda3/bin/python
version: 3.13.11 | packaged by Anaconda, Inc. | (main, Dec 10 2025, 21:28:48) [GCC 14.3.0]
torch: import failed: ModuleNotFoundError("No module named 'torch'")
cv2: import failed: ModuleNotFoundError("No module named 'cv2'")
numpy: import failed: ModuleNotFoundError("No module named 'numpy'")
scipy: import failed: ModuleNotFoundError("No module named 'scipy'")
evo: import failed: ModuleNotFoundError("No module named 'evo'")
viser: import failed: ModuleNotFoundError("No module named 'viser'")
torch details failed: ModuleNotFoundError("No module named 'torch'")
cuda_ba: import failed: ModuleNotFoundError("No module named 'cuda_ba'")
JSON_RESULT={"python": "/home/dovetao/miniconda3/bin/python", "version": "3.13.11 | packaged by Anaconda, Inc. | (main, Dec 10 2025, 21:28:48) [GCC 14.3.0]", "imports": {"torch": {"ok": false, "error": "ModuleNotFoundError(\"No module named 'torch'\")"}, "cv2": {"ok": false, "error": "ModuleNotFoundError(\"No module named 'cv2'\")"}, "numpy": {"ok": false, "error": "ModuleNotFoundError(\"No module named 'numpy'\")"}, "scipy": {"ok": false, "error": "ModuleNotFoundError(\"No module named 'scipy'\")"}, "evo": {"ok": false, "error": "ModuleNotFoundError(\"No module named 'evo'\")"}, "viser": {"ok": false, "error": "ModuleNotFoundError(\"No module named 'viser'\")"}}, "torch": {}, "cuda_ba": {"ok": false, "error": "ModuleNotFoundError(\"No module named 'cuda_ba'\")"}, "torch_error": "ModuleNotFoundError(\"No module named 'torch'\")"}
```
