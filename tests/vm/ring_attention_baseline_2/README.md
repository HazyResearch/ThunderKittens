### Setup

```bash
conda create --name ringattn python=3.12 -y
conda activate ringattn
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 # use nightly version that supports B200s
pip install packaging ninja # required by flash attention
git clone https://github.com/Dao-AILab/flash-attention # use >=2.6.0
cd flash-attention
python setup.py install
pip install "diffusers>=0.31.0"
pip install "xfuser==0.4.3.post3"
```

### Run

Check correctness:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 TORCH_CUDA_ARCH_LIST=Blackwell OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=gpu check_correct.py
```

Benchmark:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 TORCH_CUDA_ARCH_LIST=Blackwell OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=gpu benchmark.py
```
