install:
	python -m pip install -r requirements.txt
train:
	python scripts/train.py
test:
	python -m pytest -q
run:
	python app.py
