# Accelerate Model Parallel Deep Learning Training Using Effective Graph Traversal Order in Device Placement

Source code for the paper "Accelerate Model Parallel Deep Learning Training Using Effective Graph Traversal Order in Device Placement". The paper is currently **under submission**, but arXiv preprint is available at https://arxiv.org/abs/2201.09676.

---

This folder holds the code for experiments on studying the impact of graph travesal order on device placement.
The code is based on the implementation of \[[Mitropolitsky, Milko, Zainab Abbas, and Amir H. Payberah. "Graph Representation Matters in Device Placement." Proceedings of the Workshop on Distributed Infrastructures for Deep Learning. 2020.](https://github.com/mmitropolitsky/device-placement)\]


## Summary of the repo:
- **benchmark_runner**:
  - benchmark_runner.py: BenchmarkRunner class for launching device placement experiments.
- **config**: 
  - Configurations for different experiments.
- **datasets**:
  - Instructions for downloading the dataset can be found [here](https://github.com/aravic/generalizable-device-placement/tree/master/datasets#readme)
  - *cifar10*: 32 computation graphs of convolutional neural network for image classification tasks.
  - *nmt*: 32 computation graphs of encoder-decoder networks with attention structures.
  - *ptb*: 32 computation graphs for language modeling tasks.
- **model**:
  - device placement model.
- **sim**:
  - The simlutaor used to estimate the excution time of a given placement strategy.
- Dockerfile: Dockerfile for building the Docker image for the experiments.
- README.md: README file of the repo.
- requriements.txt: Required libraries for executing the experiments.
- run_experiments.py: Python script for running experiments.


---

## Command to build and run Docker:
- Building Docker images from a Dockerfile and a “context”:
        
        # CPU-only
        sudo docker build -t placeto-cpu:v0.0 ./
        
        # CPU and GPU
        sudo docker build -t placeto-gpu:v0.0 ./

- Running a bash in a new container:
        
        # CPU-only
        sudo docker run -it -v ~/device_placement/Placeto_order:/placeto -v ~/datasets/Placeto/:/placeto/datasets placeto-cpu:v0.0 bash

        # CPU and GPU
        sudo docker run -it --gpus all  -v ~/device_placement/Placeto_order:/placeto -v ~/datasets/Placeto/:/placeto/datasets placeto-gpu:v0.0 bash

---