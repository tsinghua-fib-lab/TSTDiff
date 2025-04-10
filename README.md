# TSTDiff
This repo is for TSTDiff: Transferable Spectral-Temporal Diffusion Model for Traffic Generation

## ⌨️ Project Structure
```
Model_TSTDiff
├─code
│  │  dataset.py
│  │  diff_models_freq.py
│  │  diff_models_time.py
│  │  main_model_freq.py
│  │  main_model_time.py
│  │  run.py
│  │  run_ft.py
│  │  utils_freq.py
│  │  utils_time.py
│  │  
│  ├─result_metrics
│  │      metrix_freq.txt
│  │      metrix_time.txt
│  │      
│  └─save_model
│          
├─config
│  └─base.yaml
│      
└─data
    ├─S1   
    ├─S2
    └─S3
```

## 🔧 Envirnonment

Install Python dependencies.

```bash
conda create -n Tstdiff python==3.9
pip install -r requirements.txt
```

## Stage1: Data Preparation
Create three folders, named './data/S1', './data/S2', './data/S3', which correspond to different settings:

- S1: 5G BSs with surrounding 4G BSs, both from the previous year.
- S2: Newly deployed 5G BSs with surrounding 4G BSs, both from the current year.
- S3: Newly deployed 5G BSs with surrounding 4G BSs from the previous year.

The processed data are in .npz files. 

## Stage2: Training model
Create folder './code/save_model/', which is used to store your trained models.

### 2.1 Pre-training
- epochf, epocht: The training epochs of frequency and time model.
```bash
python run.py --epochf 50 --epocht 50
```

### 2.2 Fine-tuning
- epochf, epocht: The epochs of the pretrained frequency and time models that you want to fine-tune.
- modelfolder: Folder of pre-trained model.
- ftepf, ftept: The fine-tuning epochs of frequency and time model.
```bash
python run_ft.py --epochf 50 --epocht 50  --modelfolder foldername --ftepf 30 --ftept 30 
```

## Stage3: Evaluating
Short term test
```bash
python run_ft.py --ftepf 50 --ftept 50 --ftfolder ftfoldername
```

Long term test
```bash
python run.py --epochf 50 --epocht 50 --modelfolder foldername
```
