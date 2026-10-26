# GQL runtime dependency sets

- `../requirements.txt` contains shared FastAPI, GraphQL, and Uvicorn dependencies.
- `cuda.txt` adds the pinned PyTorch wheel from the CUDA 12.4 wheel index.
- `rocm.txt` adds the pinned PyTorch wheel from the ROCm 6.2.4 wheel index.

Install the shared file first, then exactly one backend file. CUDA and ROCm wheels are not interchangeable. The existing YOLOv7 and Caffe SSD adapters use OpenCV DNN; installing PyTorch alone does not accelerate those adapters. Validate the chosen base image, Python ABI, driver, and hardware on the target host before production use.
