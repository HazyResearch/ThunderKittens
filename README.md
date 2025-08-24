\# 🐱 ThunderKittens Docker Setup



This Docker setup provides a complete environment for developing and running ThunderKittens CUDA kernels with all necessary dependencies pre-installed.



\## 🚀 Quick Start



1\. \*\*Clone this setup\*\* (save all the provided files in a directory)

2\. \*\*Make the setup script executable:\*\*

&nbsp;  ```bash

&nbsp;  chmod +x setup.sh

&nbsp;  ```

3\. \*\*Run the setup:\*\*

&nbsp;  ```bash

&nbsp;  ./setup.sh

&nbsp;  ```



That's it! The container will be built and started automatically.



\## 📋 Prerequisites



\- \*\*Docker\*\* (with docker-compose or Docker Compose plugin)

\- \*\*NVIDIA Docker runtime\*\* (`nvidia-docker2`)

\- \*\*NVIDIA GPU\*\* with compatible drivers

\- \*\*CUDA-capable GPU\*\* (compute capability 7.0+, recommended 8.0+ for H100/A100)



\### Installing NVIDIA Docker Runtime



```bash

\# Ubuntu/Debian

distribution=$(. /etc/os-release;echo $ID$VERSION\_ID)

curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -

curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list



sudo apt-get update \&\& sudo apt-get install -y nvidia-docker2

sudo systemctl restart docker

```



\## 🎛️ Available Commands



The `setup.sh` script provides several commands for managing your ThunderKittens environment:



```bash

./setup.sh build     # Build and start the container (default)

./setup.sh start     # Start existing container  

./setup.sh stop      # Stop the container

./setup.sh restart   # Restart the container

./setup.sh shell     # Open shell in running container

./setup.sh logs      # Show container logs

./setup.sh clean     # Remove container and images

./setup.sh help      # Show help message

```



\## 🌐 Access Points



Once the container is running:



\- \*\*Jupyter Notebook\*\*: http://localhost:8888 (no password required)

\- \*\*TensorBoard\*\*: http://localhost:6006

\- \*\*Container Shell\*\*: `docker exec -it thunderkittens-dev bash`



\## 📁 Directory Structure



```

.

├── Dockerfile              # Main Docker configuration

├── docker-compose.yml      # Docker Compose configuration

├── setup.sh               # Setup and management script

├── .env                   # Environment variables (auto-generated)

├── workspace/             # Your local development files

└── examples/              # Example scripts and notebooks

```



\## 🔧 Container Details



\### Base Image

\- \*\*NVIDIA CUDA 12.1\*\* development environment on Ubuntu 22.04

\- \*\*Python 3.10\*\* with conda environment management



\### Pre-installed Software

\- \*\*ThunderKittens\*\* (latest from GitHub)

\- \*\*PyTorch\*\* with CUDA 12.1 support

\- \*\*CUDA Toolkit\*\* and development headers

\- \*\*CMake\*\*, \*\*Ninja\*\* build system

\- \*\*Jupyter\*\* notebook environment

\- \*\*Common Python packages\*\* (numpy, scipy, matplotlib, pytest)



\### GPU Access

The container is configured to access all available GPUs with full capabilities.



\## 🧪 Running Your First ThunderKittens Kernel



1\. \*\*Access the container:\*\*

&nbsp;  ```bash

&nbsp;  ./setup.sh shell

&nbsp;  ```



2\. \*\*Activate the conda environment\*\* (if not already activated):

&nbsp;  ```bash

&nbsp;  conda activate thunderkittens

&nbsp;  ```



3\. \*\*Navigate to examples:\*\*

&nbsp;  ```bash

&nbsp;  cd /workspace/ThunderKittens/examples

&nbsp;  ```



4\. \*\*Run a sample kernel:\*\*

&nbsp;  ```bash

&nbsp;  # Example: compile and run a basic kernel

&nbsp;  cd based/

&nbsp;  python test\_based.py

&nbsp;  ```



\## 📝 Development Workflow



\### Using Jupyter Notebooks

1\. Open http://localhost:8888 in your browser

2\. Navigate to `/workspace/ThunderKittens/` or `/workspace/shared/` for your files

3\. Create new notebooks or open existing examples



\### Using VS Code with Remote Containers

1\. Install the "Remote - Containers" extension

2\. Open the folder in VS Code

3\. When prompted, "Reopen in Container"



\### Direct Container Development

```bash

\# Open shell in container

./setup.sh shell



\# Your conda environment is pre-activated

\# Start developing your kernels

cd /workspace/ThunderKittens

```



\## 🎯 Common Use Cases



\### 1. Kernel Development

```bash

\# Create a new kernel file

cd /workspace/shared

nano my\_kernel.cu



\# Compile with nvcc

nvcc -arch=sm\_80 -o my\_kernel my\_kernel.cu -I/workspace/ThunderKittens/include

```



\### 2. Python Binding Development

```bash

\# Install your kernel as a Python module

cd /workspace/ThunderKittens

pip install -e .

```



\### 3. Performance Benchmarking

```bash

\# Run built-in benchmarks

cd /workspace/ThunderKittens/benchmarks

python attention\_benchmark.py

```



\## 🐛 Troubleshooting



\### GPU Not Accessible

```bash

\# Test GPU access in container

docker exec -it thunderkittens-dev nvidia-smi

```



If this fails:

1\. Check NVIDIA Docker installation

2\. Restart Docker daemon: `sudo systemctl restart docker`

3\. Verify GPU drivers: `nvidia-smi` on host



\### Build Issues

```bash

\# Clean rebuild

./setup.sh clean

./setup.sh build

```



\### Memory Issues

If you encounter CUDA out of memory errors:

```bash

\# Set GPU memory limit in .env file

echo "CUDA\_MEM\_LIMIT=8G" >> .env

./setup.sh restart

```



\## 📚 Additional Resources



\- \[ThunderKittens GitHub Repository](https://github.com/HazyResearch/ThunderKittens)

\- \[ThunderKittens Documentation](https://github.com/HazyResearch/ThunderKittens/tree/main/docs)

\- \[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)



\## 🤝 Contributing



To contribute to ThunderKittens:

1\. Fork the repository

2\. Make changes in your local `workspace/` directory

3\. Test in the container environment

4\. Submit pull requests upstream



\## 📄 License



This Docker setup is provided as-is. ThunderKittens itself is subject to its own license terms.

