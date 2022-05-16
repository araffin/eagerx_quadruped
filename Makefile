# Adapted from DLR-RM/stable-baselines3
PACKAGE_NAME=eagerx_quadruped

SHELL=/bin/bash
LINT_PATHS=${PACKAGE_NAME}/ *.py tests/

pytest:
	bash ./scripts/run_tests.sh

type:
	pytype ${PACKAGE_NAME}/

check-codestyle:
	# Sort imports
	isort --check ${LINT_PATHS}
	# Reformat using black
	black --check ${LINT_PATHS}

format: codestyle

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

commit-checks: codestyle type lint

.PHONY: check-codestyle
