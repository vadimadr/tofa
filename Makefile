.PHONY: clean clean-test clean-pyc clean-build docs

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

BROWSER := bash -c 'open $$0 || xdg-open $$0 || sensible-browser $$0 || x-www-browser $$0'

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -fr .mypy_cache
	rm -f coverage.xml
	rm -f junit.xml

lint: ## check style with flake8
	poetry run flake8 tofa tests
	poetry run mypy tofa tests

format:
	poerty run autoflake -ir tofa
	poerty run isort .
	poerty run black .

test: ## run tests quickly with the default Python
	poetry run pytest

coverage-report: ## check code coverage quickly with the default Python
	poetry run pytest --cov=tofa --junitxml=junit.xml
	poetry run coverage html
	poetry run coverage xml

coverage: coverage-report
	$(BROWSER) htmlcov/index.html

publish: dist ## package and upload a release
	poetry publish

dist: clean ## builds source and wheel package
	poetry build

develop: clean ## Conifure python environment for develooping this package
	poetry install --with dev
