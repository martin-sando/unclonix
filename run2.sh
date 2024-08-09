#!/bin/bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd key_generation_sandbox
date
../.venv/bin/python3 bloblist_operations.py $*
date
