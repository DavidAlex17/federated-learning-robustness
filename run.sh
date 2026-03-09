#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
cd ui
PYTHONPATH=.. python app.py
