# CPU
docker compose -f gpu/compose.yaml up --build gql-cpu

# NVIDIA CUDA
docker compose -f gpu/compose.yaml --profile cuda up --build gql-cuda

# AMD ROCm
docker compose -f gpu/compose.yaml --profile rocm up --build gql-rocm

jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
