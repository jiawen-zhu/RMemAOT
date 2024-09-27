# RMemAOT for VOTSt2024 challenge

RMemAOT for VOTSt2024 ranked **1st** in the [**VOTSt 2024**](https://votchallenge.net/vots2024/) challenge ([leaderboard](https://eu.aihub.ml/competitions/254#results)).

Raw results can be download from [here](https://1drv.ms/u/c/72bf835b6b16ea56/EbkCCUCYR5dJl5qIKKSG4eoBXuwy-gISkHZnXYZuuRnLVA?e=lPcqgs)(OneDrive).

## Install dependencies
* Create and activate a conda environment 
```bash
conda create -n RMemAOT python=3.8
conda activate RMemAOT
```  
* Install PyTorch
```bash
conda install -c pytorch pytorch=1.12.1 torchvision=0.13.1
```  

* Install packages
```bash
pip install -r requirements.txt
```  

## Model Download
- Pretrained model can be downloaded from [here](https://github.com/yoxu515/aot-benchmark/blob/main/MODEL_ZOO.md).
- Put the pretrained weight in `.aot_plus/pretrain_models/`.

## Run the tracker
* Enter the vot workspace root
```
cd ./RMemAOT/aot_plus/votst24_test
```  
* Initialize the vot workspace
```
vot initialize vots2024/votst.yaml
```
* Copy our trackers.ini to your vot workspace, config the path in tracker.ini 
```
cp /path/to/our/trackers.ini /path/to/votst24_test/trackers.ini
```
* Config the palette_template path in run_tracker_vot.py (line338)

* Then you can test RMemAOT on votst2024 
```
vot evaluate RMemAOT
vot pack RMemAOT
```

## Acknowledgement
This repo is built upon [RMem](https://github.com/Restricted-Memory/RMem).
Thanks for the excellent work.

