

Here we document common issues:

If you can't find ```nvcc```, or you experience issues where your environment is pointing to the wrong CUDA version:
```
export CUDA_HOME=/usr/local/cuda-12/
export PATH=${CUDA_HOME}/bin:${PATH} 
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
```

