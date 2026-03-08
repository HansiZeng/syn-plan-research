#!/bin/bash

MODEL_NAME=${1:-"Qwen/Qwen3-32B"}

# 设置并行大小（默认每个模型 4 张卡）
tensor_parallel_size=4

echo "Serving primary model: $MODEL_NAME"
echo "tensor_parallel_size: $tensor_parallel_size"

# 构建镜像
docker_name=qwen3-server
docker build -f docker/Dockerfile.qwen3 -t $docker_name .

# 创建 Docker 网络（如果不存在）
docker network inspect llm-net >/dev/null 2>&1 || docker network create llm-net

echo start serving models...

# Serve 主模型（前 4 张卡：0,1,2,3）
docker run --gpus '"device=0,1,2,3"' --rm -it \
  --network llm-net \
  --name main-model-server \
  -p 8980:8980 \
  --shm-size=32g \
  -v /home/hanszeng/WebThinker:/workspace/ \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  $docker_name \
  bash -c "cd /workspace && \
    VLLM_LOGGING_LEVEL=DEBUG vllm serve ${MODEL_NAME} \
    --tensor-parallel-size $tensor_parallel_size \
    --gpu-memory-utilization 0.8 \
    --port 8980 \
    --dtype auto \
    --api-key token-abc123"