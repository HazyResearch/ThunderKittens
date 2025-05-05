Howdy!

Install this repo with:

```bash

pip install -e .

```

To generate tokens with the normal PyTorch model:


```bash

python kvm_batch_runner/generate.py mode=model ntok=30 prompt="tell me a funny joke about cookies"

```

To use the Python VM:

```bash

python kvm_batch_runner/generate.py mode=pyvm ntok=30 prompt="tell me a funny joke about cookies"

```

To use the kvm:

```bash

python v/generate.py mode=kvm ntok=30 prompt="tell me a funny joke about cookies"

```