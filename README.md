# GVPermformer

A Graph Vision Transformer with Permutation-based Architecture.

## Project Structure

```
GVPermformer/
├── configs/           # Configuration files
├── src/              # Source code
├── scripts/          # Utility scripts
├── training_data/    # Training data
├── tests/            # Test files
└── requirements.txt  # Project dependencies
```

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Development

This project follows:

- PEP 8 style guide
- Type hints (checked with mypy)
- Black code formatting
- Flake8 linting
- isort import sorting

To run code quality checks:

```bash
black .
isort .
flake8
mypy .
```

## Testing

Run tests with pytest:

```bash
pytest
```

## License

See the LICENSE file for details.
