install:
	pip install -r requirements.txt

train:
	python src/train.py

test:
	pytest tests/ -v

lint:
	flake8 src/ --max-line-length=100

all: install lint test train
