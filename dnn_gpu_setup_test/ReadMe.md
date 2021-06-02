# Setup & Test GPU conda env for Win10/Anaconda3

- As of 2020-March, access to GPU is not supported by WSL, which drives the need of setup GPU env natively (Linux VM or dual OS are certainly alternatives, but not considered)

- The batch script of this repo is to do so with native windows command (on top of Anaconda3 prompt)

- It should create a new conda env with below setup:
    - Python=3.7.6
    - Tensorflow-gpu=2.0.0
    - PyTorch=1.4.0
    - And verify if GPU is ready for use

- More python pacakes can be added into the setup script if required

- Expect outcome from conda_dnn_gpu_setup.bat:
    - tensorflow-gpu: True
    - torch: True