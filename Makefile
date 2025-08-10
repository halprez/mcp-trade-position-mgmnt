.PHONY: help install dev test lint format clean run docker

help:
	@echo "TPM MCP Server - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  install     Install dependencies with uv"
	@echo "  dev         Install with dev dependencies"
	@echo ""
	@echo "Development:"
	@echo "  run         Start the MCP server"
	@echo "  test        Run tests with pytest"
	@echo "  lint        Run ruff linter"
	@echo "  format      Format code with black and ruff"
	@echo "  typecheck   Run mypy type checking"
	@echo "  pre-commit  Run all pre-commit hooks"
	@echo ""
	@echo "Data:"
	@echo "  sample-data Generate sample dataset for testing"
	@echo "  process-data Process dataset into database"
	@echo ""
	@echo "Docker:"
	@echo "  docker      Build and run with docker-compose"
	@echo "  docker-dev  Run docker-compose for development"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean       Remove cache files and artifacts"

install:
	uv sync

dev:
	uv sync --extra dev --extra notebook
	uv run pre-commit install

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run black .
	uv run ruff check --fix .

typecheck:
	uv run mypy src/

run:
	uv run uvicorn src.main:app --reload --port 8000

sample-data:
	uv run python scripts/download_data.py --sample

process-data:
	uv run python scripts/process_data.py

process-sample:
	uv run python scripts/process_data.py --sample

docker:
	docker-compose up --build

docker-dev:
	docker-compose -f docker-compose.yml -f deployment/docker-compose.yml up --build

pre-commit:
	uv run pre-commit run --all-files

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +