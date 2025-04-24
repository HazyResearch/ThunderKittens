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

Op-level testing:

```bash

python kvm_runner/test_kvm.py stop_after_op=qkv
python kvm_runner/test_kvm.py stop_after_op=partial_attn start_after_op=qkv
python kvm_runner/test_kvm.py stop_after_op=attn_reduction start_after_op=partial_attn
python kvm_runner/test_kvm.py stop_after_op=o_proj start_after_op=attn_reduction
python kvm_runner/test_kvm.py stop_after_op=up_gate start_after_op=o_proj
python kvm_runner/test_kvm.py stop_after_op=down_proj start_after_op=up_gate


```