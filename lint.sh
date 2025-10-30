echo "\nLinting with ruff"
poetry run ruff check . --fix

echo "Formatting with black"
poetry run black . --color

echo "\nFormatting with isort"
poetry run isort .

echo "\nType checking with mypy"
poetry run mypy --config-file=pyproject.toml --python-executable /Users/matt/Library/Caches/pypoetry/virtualenvs/hatchet-python-quickstart-tNofGvRK-py3.13/bin/python
