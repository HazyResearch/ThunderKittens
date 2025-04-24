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

```