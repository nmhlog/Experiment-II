# Aktifasi Pods
kubectl apply pods-name

# Masuk kedalam container
kubectl exec -it gpu-03-naufalmh-cudasdk -- /bin/bash

# Compiling Code.
nvcc -o output output.cu

# if using cublas
nvcc -o matmul_cublas -lcublas matmul_cublas.cu
