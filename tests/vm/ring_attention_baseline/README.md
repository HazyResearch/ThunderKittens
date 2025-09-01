# H100

Tested on Python 3.12, NVIDIA GPU driver 570.124.06, CUDA version 12.8

### Setup

```bash
conda create --name ringattn python=3.12
conda activate ringattn
pip install -r requirements.H100.txt
```

### Run

```bash
CUDA_VISIBLE_DEVICES=... python baseline.py
```

# B200

Tested on Python 3.12, NVIDIA GPU driver 570.133.07, CUDA version 12.8

### Setup

```bash
conda create --name ringattn python=3.12
conda activate ringattn
pip install -r requirements.B200.txt
```

### Run

```bash
CUDA_VISIBLE_DEVICES=... python baseline.py
```
