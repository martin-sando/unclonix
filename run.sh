#!/bin/bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd key_generation_sandbox
../.venv/bin/python3 key_generation_sandbox.py
