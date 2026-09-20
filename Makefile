.PHONY: install lint format test bench demo clean

install:
	pip install -e .[dev]
	-pre-commit install

lint:
	ruff check src bench tests
	mypy src bench

format:
	black src bench tests
	ruff check --fix src bench tests

test:
	pytest tests/

bench:
	python -m bench.bench_latency --config configs/experiments/smoke.yaml --duration 10

demo:
	@echo "Tap-to-select demo. Mock mode, no camera/mic:"
	@echo "  python demo/tap_to_select.py --config configs/experiments/dolphin_tap.yaml"
	@echo "For --live (real camera/mic/speakers), get explicit permission first."
	@echo "Needs a GUI-capable OpenCV build: pip uninstall -y opencv-python-headless && pip install -e \".[demo]\""

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/
