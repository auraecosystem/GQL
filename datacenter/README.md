# Datacenter deployment

The Compose example intentionally binds the API to localhost. Put it behind an authenticated reverse proxy before remote access. This starter records job requests in memory and does not execute training or inference. CUDA and ROCm host/device compatibility must be validated on the target system.
