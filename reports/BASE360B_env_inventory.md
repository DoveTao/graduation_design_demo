# BASE360B env inventory

## Existing pytorch env check
- `existing_pytorch_env_checked = true`
- `existing_pytorch_env_usable = false`
- `torch_import_in_pytorch_env = pass`
- `cuda_available_in_pytorch_env = true`
- `cv2_import_in_pytorch_env = pass`
- `evo_import_in_pytorch_env = pass`
- `viser_import_in_pytorch_env = pass`
- `selected_runtime_env = blocked`

## Python / CUDA
- python executable: `/home/dovetao/miniconda3/envs/pytorch/bin/python`
- python version: `3.12.12`
- torch: `2.9.1+cu128`
- torch.cuda.is_available(): `True`
- torch.version.cuda: `12.8`
- GPU: `NVIDIA GeForce RTX 3060 Laptop GPU`
- cv2: `4.13.0`
- scipy: `1.17.1`
- evo: `1.36.4`
- viser: `1.0.27`

## Official repo check
- official repo: `/tmp/360DVO_official`
- `environment.yml`: present
- README install steps checked: `conda env create -f environment.yml`, `wget eigen-3.4.0.zip`, `pip install .`, download `360dvo.pth`
- demo.py interface checked: `--network --imagedir --stride --skip --config --timeit --viz --save_trajectory --save_ply --plot --name`
- expected default weight path: `/tmp/360DVO_official/360dvo.pth`

## Install attempts
- pytorch env pip repair log: `/home/dovetao/graduation_design_demo/logs/base360b/pytorch_env_pip_install.log`
- official pip install log: `/home/dovetao/graduation_design_demo/logs/base360b/official_pip_install.log`
- official pip install no-build-isolation log: `/home/dovetao/graduation_design_demo/logs/base360b/official_pip_install_nobi.log`
- official pip install arch86 log: `/home/dovetao/graduation_design_demo/logs/base360b/official_pip_install_arch86.log`
- official env create log: `/home/dovetao/graduation_design_demo/logs/base360b/official_env_create.log`
- manual env create log: `/home/dovetao/graduation_design_demo/logs/base360b/manual_env_create.log`
- current blocker in pytorch env: `pip install .` reaches CUDA extension build and fails on Torch API incompatibility in `correlation_kernel.cu`.
- manual `base360dvo` env creation was started but had not completed at the time of this report snapshot.
