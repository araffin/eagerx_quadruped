# Adapted from DLR-RM/stable-baselines3
PACKAGE_NAME=eagerx_ode

SHELL=/bin/bash
LINT_PATHS=${PACKAGE_NAME}/ *.py

pytest:
	bash ./scripts/run_tests.sh

check-codestyle:
	# Sort imports
	isort --check ${LINT_PATHS}
	# Reformat using black
	black --check ${LINT_PATHS}

codestyle:
	# Sort imports
	isort ${LINT_PATHS}
	# Reformat using black
	black ${LINT_PATHS}

lint:
	# stop the build if there are Python syntax errors or undefined names
	# see https://lintlyci.github.io/Flake8Rules/
	flake8 ${LINT_PATHS} --count --select=E9,F63,F7,F82 --show-source --statistics
	# exit-zero treats all errors as warnings.
	flake8 ${LINT_PATHS} --count --exit-zero --statistics

.PHONY: check-codestyle
