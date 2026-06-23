# CarSpec AI Makefile
.PHONY: setup train deploy test clean help

help:
	@echo "CarSpec AI 命令:"
	@echo "  make setup    - 安装依赖并准备数据"
	@echo "  make train    - 训练所有模型"
	@echo "  make run      - 启动Web应用"
	@echo "  make deploy   - 部署到HuggingFace Spaces"
	@echo "  make clean    - 清理缓存和临时文件"

setup:
	pip install -r requirements.txt
	python -m scripts.make_dataset

train:
	python setup.py

run:
	python main.py

deploy:
	@echo "部署到 HuggingFace Spaces..."
	@echo "1. 创建 Space: https://huggingface.co/new-space"
	@echo "2. 选择 Docker SDK"
	@echo "3. 推送代码: git push space main"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
