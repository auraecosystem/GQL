# GQL runtime dependency sets

- `../requirements.txt` is the shared API/control-plane and OpenCV runtime used by CPU images.
- `cuda.txt` adds the pinned PyTorch wheel for the CUDA 12.4 image.
- `rocm.txt` adds the pinned PyTorch wheel for the ROCm 6.2.4 wheel index.

CUDA and ROCm wheels are intentionally separate and must not be combined in one environment. The current image adapters still use OpenCV DNN for YOLOv7/Caffe SSD; installing PyTorch does not automatically make those adapters GPU accelerated. Confirm the selected base image, Python ABI, driver, GPU, and framework wheel compatibility on the target host before production use.
