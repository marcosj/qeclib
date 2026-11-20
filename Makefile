.PHONY: run docs

run:
	PYTHONPATH=src python3 src/main.py

docs:
	PYTHONPATH=src pdoc --math --docformat google -o docs/api src/*.py
