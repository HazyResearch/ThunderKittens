# Setup Instructions with Conda

To create a new conda env compatible with kittens, use:

```bash
conda create -n kittens python=3.11
conda activate kittens
conda install cuda==12.4.0 -c nvidia
```

Note that if you try to profile your kernels by launching `ncu` or `ncu-ui`, you may get an error message like:

```
ERROR : nsight-compute directory is not found under <YOUR_PATH_HERE>/miniconda3/envs/kittens/bin/../ or /opt/nvidia. Nsight Compute is not installed on your system.
```

This seems to be due to a bug in the ncu launcher script's detection of the appropriate programs when they're installed under conda. As a workaround, adding the following shell functions to your `.bashrc` / `.zshrc` should do the trick:

```bash
ncu2() {
    local original_ncu_path=$(which ncu)
    local nsight_compute_base_dir=$(dirname $(dirname $original_ncu_path))/nsight-compute
    local nsight_compute_version_dir=$(find $nsight_compute_base_dir -mindepth 1 -maxdepth 1 -type d | head -n 1)

    if [ -n "$nsight_compute_version_dir" ]; then
        export LATEST_NSIGHT_COMPUTE_TOOL_DIR="$nsight_compute_version_dir"
    else
        echo "Nsight Compute directory not found."
        return 1
    fi

    # Invoke the original ncu command with all passed arguments
    sudo LATEST_NSIGHT_COMPUTE_TOOL_DIR="$nsight_compute_version_dir" $original_ncu_path "$@"
}


ncu-ui2() {
    local original_ncu_ui_path=$(which ncu-ui)
    local nsight_compute_base_dir=$(dirname $(dirname $original_ncu_ui_path))/nsight-compute    
    local nsight_compute_version_dir=$(find $nsight_compute_base_dir -mindepth 1 -maxdepth 1 -type d | head -n 1)

    if [ -n "$nsight_compute_version_dir" ]; then
        export LATEST_NSIGHT_COMPUTE_TOOL_DIR="$nsight_compute_version_dir"
    else
        echo "Nsight Compute directory not found."
        return 1
    fi

    # Invoke the original ncu command with all passed arguments
    sudo LATEST_NSIGHT_COMPUTE_TOOL_DIR="$nsight_compute_version_dir" $original_ncu_ui_path "$@"
}
```

With these commands, you could be able to use `ncu2` and `ncu-ui2` as replacements for `ncu` and `ncu-ui`, respectively.