FROM python:3.9-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖（不含 torch，减小镜像体积）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 创建必要目录
RUN mkdir -p models data/raw data/processed data/outputs

# 端口
ENV PORT=7860
EXPOSE 7860

# 启动
CMD ["python", "main.py"]
