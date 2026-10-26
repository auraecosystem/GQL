# GQL runtime dependency sets

- `../requirements.txt` provides the shared API/control-plane and OpenCV runtime used by CPU images.
- `cuda.txt` adds the pinned PyTorch wheel for the CUDA 12.4 image.
- `rocm.txt` adds the pinned PyTorch wheel for the ROCm 6.2.4 wheel index.

CUDA and ROCm wheels are intentionally separate and must not be combined in one environment. Existing YOLOv7 and Caffe SSD adapters use OpenCV DNN; installing PyTorch does not automatically accelerate them. Confirm base-image, Python ABI, driver, GPU, and framework-wheel compatibility on the target host before production use.
