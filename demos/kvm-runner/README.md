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

```