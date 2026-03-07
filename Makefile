setup:
	pip install -r requirements.txt

batch:
	PYTHONPATH=. python experiments/run_batch.py

smoke:
	PYTHONPATH=. python experiments/run_smoke.py --run-id smoke --clean

test:
	PYTHONPATH=. python -m pytest -q

clean:
	rm -rf experiments/results/*/ experiments/plots/*/ 2>/dev/null || true
