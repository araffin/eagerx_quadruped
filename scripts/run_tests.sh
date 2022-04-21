#!/bin/bash
python3 -m pytest --cov-config .coveragerc tests/ --cov-report html --cov-report xml --cov-report term --cov=. -v --color=yes
