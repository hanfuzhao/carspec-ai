.PHONY: setup train deploy test clean help

help:
	@echo "CarSpec AI commands:"
	@echo "  make setup    - Install dependencies and prepare data"
	@echo "  make train    - Train all models"
	@echo "  make run      - Start web app"
	@echo "  make deploy   - Deploy to HuggingFace Spaces"
	@echo "  make clean    - Clean cache and temporary files"

setup:
	pip install -r requirements.txt
	python -m scripts.make_dataset

train:
	python setup.py

run:
	python main.py

deploy:
	@echo "Deploying to HuggingFace Spaces..."
	@echo "1. Create Space: https://huggingface.co/new-space"
	@echo "2. Select Docker SDK"
	@echo "3. Push code: git push space main"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
