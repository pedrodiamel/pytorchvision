# Contributing to PytorchVision

## Style code

Changes to Python code should conform to [Black Code Style Guide](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)

The Python project code should be formatted by Black, and checked against Pep8 compliance with flake8. Instead of relying directly on black, however, we rely on Ufmt, for compatibility with CI infrastructure.

### Development installation

    docker-compose up --build -d
    docker-compose down
    docker exec -it pytv-dev /bin/bash

### Install

    pip install flake8 typing mypy pytest pytest-mock
    pip install ufmt==1.3.2 black==22.3.0 usort==1.0.2
    pip install pre-commit

### Code formatting

To format your code, install ufmt with pip install ufmt==1.3.2 black==22.3.0 usort==1.0.2 and use e.g.:

    ufmt format pytvision

For the vast majority of cases, this is all you should need to run. For the formatting to be a bit faster, you can also choose to only apply ufmt to the files that were edited in your PR with e.g.:

    ufmt format `git diff main --name-only`

### Pre-commit hooks

For convenience and purely optionally, you can rely on pre-commit hooks, which will run both ufmt and flake8 prior to every commit.

First, install the pre-commit package with pip install pre-commit, and then run pre-commit install at the root of the repo for the hooks to be set up - that's it.

    pip install pre-commit
    pre-commit install

Read the pre-commit docs to learn more and improve your workflow. You'll see, for example, that pre-commit run --all-files will run both ufmt and flake8 without the need for you to commit anything and that the --no-verify flag can be added to git commit to temporarily deactivate the hooks.

### Type annotations

The codebase has type annotations, please make sure to add type hints if required. We use mypy tool for type checking:

    mypy --config-file mypy.ini

Unit tests

If you have modified the code by adding a new feature or a bug-fix, please add unit tests for that. To run a specific test:

    pytest test/<test-module.py> -vvv -k <test_myfunc>
    # e.g. pytest test/test_transforms.py -vvv -k test_center_crop

If you would like to run all tests:

    pytest test -vvv

### Documentation

Python projects should be [Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for formatting docstrings. Length of line inside docstrings block must be limited to 120 characters.
