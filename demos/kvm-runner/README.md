Howdy!

Install this repo with:

```bash

pip install -e .

```

To generate tokens with the normal PyTorch model:


```bash

python kvm_runner/generate.py mode=model ntok=30 prompt="tell me a funny joke about cookies"

```

To use the Python VM:

```bash

python kvm_runner/generate.py mode=pyvm ntok=30 prompt="tell me a funny joke about cookies"

```

To use the kvm:

```bash

python kvm_runner/generate.py mode=kvm ntok=30 prompt="tell me a funny joke about cookies"

```

Op-level testing:

```bash

# individual ops
python kvm_runner/test_kvm.py stop_after_op=qkv
python kvm_runner/test_kvm.py stop_after_op=partial_attn start_after_op=qkv
python kvm_runner/test_kvm.py stop_after_op=attn_reduction start_after_op=partial_attn
python kvm_runner/test_kvm.py stop_after_op=o_proj start_after_op=attn_reduction
python kvm_runner/test_kvm.py stop_after_op=up_gate start_after_op=o_proj
python kvm_runner/test_kvm.py stop_after_op=down_proj start_after_op=up_gate

# cumulative
python kvm_runner/test_kvm.py stop_after_op=qkv
python kvm_runner/test_kvm.py stop_after_op=partial_attn
python kvm_runner/test_kvm.py stop_after_op=attn_reduction
python kvm_runner/test_kvm.py stop_after_op=o_proj
python kvm_runner/test_kvm.py stop_after_op=up_gate
python kvm_runner/test_kvm.py stop_after_op=down_proj


# whole layers
python kvm_runner/test_kvm.py layer_override=1
python kvm_runner/test_kvm.py layer_override=2

python kvm_runner/test_kvm.py layer_override=100 stop_after_op=qkv


# loop ops

python kvm_runner/test_kvm.py stop_after_op=down_proj start_after_op=up_gate skip_pyvm=T reps=100
python kvm_runner/test_kvm.py stop_after_op=up_gate start_after_op=o_proj skip_pyvm=T reps=100
python kvm_runner/test_kvm.py stop_after_op=o_proj start_after_op=attn_reduction skip_pyvm=T reps=100

python kvm_runner/test_kvm.py stop_after_op=down_proj start_after_op=up_gate skip_pyvm=T instruction_reps=100 exec_reps=1000 skip_starting_instructions=T
python kvm_runner/test_kvm.py stop_after_op=up_gate start_after_op=o_proj skip_pyvm=T instruction_reps=100 exec_reps=1000 skip_starting_instructions=T
python kvm_runner/test_kvm.py stop_after_op=o_proj start_after_op=attn_reduction skip_pyvm=T instruction_reps=100 exec_reps=1000 skip_starting_instructions=T


python kvm_runner/test_kvm.py stop_after_op=qkv
python kvm_runner/test_kvm.py stop_after_op=partial_attn start_after_op=qkv
python kvm_runner/test_kvm.py stop_after_op=attn_reduction start_after_op=partial_attn
python kvm_runner/test_kvm.py stop_after_op=o_proj start_after_op=attn_reduction
python kvm_runner/test_kvm.py stop_after_op=up_gate start_after_op=o_proj
python kvm_runner/test_kvm.py stop_after_op=down_proj start_after_op=up_gate

python kvm_runner/test_kvm.py stop_after_op=o_proj start_after_op=attn_reduction skip_pyvm=T instruction_reps=1000 exec_reps=1000000000 barrier_init_val=100000 skip_starting_instructions=T


python kvm_runner/test_kvm.py skip_pyvm=T layer_limit=None barrier_init_val=100000 exec_reps=100000 diff_tensors=F


python kvm_runner/test_kvm.py stop_after_op=qkv instruction_reps=20 skip_pyvm=T 

python kvm_runner/test_kvm.py skip_pyvm=T layer_limit=None

# lm head testing
python kvm_runner/test_kvm.py layer_limit=None

python kvm_runner/generate.py mode=kvm ntok=30 prompt="tell me a funny joke about cookies"

python kvm_runner/test_kvm.py layer_limit=None skip_pyvm=T outfile=timings.pt

# testing 8 consumer warps

python kvm_runner/test_kvm.py sched=wave .full outfile=wave.pkl

```

vllm / sglang testing:

```bash

conda create -n vllm-bench -y python=3.12
conda activate vllm-bench
conda install -y nvidia/label/cuda-12.4.1::cuda-toolkit
pip install uv
uv pip install vllm==0.8.5.post1
uv pip install flashinfer-python==0.2.2.post1 -i https://flashinfer.ai/whl/cu124/torch2.5 --no-deps

conda create -n sgl-bench -y python=3.12
conda activate sgl-bench
conda install -y nvidia/label/cuda-12.4.1::cuda-toolkit
pip install uv
uv pip install 'sglang[all]'==0.4.6.post2


ca vllm-bench
vllm serve meta-llama/Llama-3.2-1B-Instruct

python -m sglang.launch_server --model-path meta-llama/Llama-3.2-1B-Instruct --port 8000 --max-running-requests 1 --enable-torch-compile


https://github.com/sgl-project/sglang/blob/7051985ebc48ed16556338fd7e8f35a104c626d3/docker/Dockerfile.blackwell

# uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
# uv pip install https://github.com/sgl-project/whl/releases/download/v0.0.8.post1/sgl_kernel-0.0.8.post1+cu128-cp39-abi3-manylinux2014_x86_64.whl
# uv pip install setuptools==75.0.0 wheel==0.41.0 scikit-build-core
# git clone --depth=1 https://github.com/sgl-project/sglang.git
# cd sglang 
# uv pip install -e "python[blackwell]"


# RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# RUN pip3 install https://github.com/sgl-project/whl/releases/download/v0.1.2/sgl_kernel-0.1.2+cu128-cp39-abi3-manylinux2014_x86_64.whl \
#     && pip3 install setuptools==75.0.0 wheel==0.41.0 scikit-build-core

# RUN git clone --depth=1 https://github.com/sgl-project/sglang.git \
#     && cd sglang && pip3 install -e "python[blackwell]"

# RUN pip3 install nvidia-nccl-cu12==2.26.2.post1 --force-reinstall --no-deps

uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
uv pip install https://github.com/sgl-project/whl/releases/download/v0.1.2/sgl_kernel-0.1.2+cu128-cp39-abi3-manylinux2014_x86_64.whl
uv pip install setuptools==75.0.0 wheel==0.41.0 scikit-build-core
git clone --depth=1 https://github.com/sgl-project/sglang.git
cd sglang 
uv pip install -e "python[blackwell]"

```

```python

from openai import OpenAI
client = OpenAI(
    api_key='fake-key',
    base_url="http://0.0.0.0:8000/v1"
)
def go(t=100):
    return client.completions.create(
        model="meta-llama/Llama-3.2-1B-Instruct",
        prompt="a a a a a a a a a a",
        max_tokens=t,
        temperature=0,
        n=1,
    )

```