# VR Teleoperation Library

The current readme is for collaborators:
## Installation
### Virtual environment
Install a virtual environment with Python=3.11
```bash
mamba create -n move_to_server python=3.11
mamba activate move_to_server
```

### Install Poetry and install dependencies via Poetry
```bash
mamba install poetry
poetry install
```

### Install pre-commit 
```bash
pre-commit install
```

## CI
There are two CI checks in place:

1. Static type checking: as dynamic testing for robots are hard, it is important to use static type checking to ensure everything is in place. Run `mypy --strict .` to check types and make sure you annotate types.
2. Pre-commit: pre-commit makes sure that the code format is consistent. After `pre-commit install`, it will run automatically every time you do `git commit`. If reformatting happens, the `git commit` will fail, but it should work after you `git add` the reformatted files. In rare conditions, `ruff` will fail to reformat, but it will print out why.

Make sure to keep PRs slim and only solving one problem at a time. 

Happy coding! Contact Hao Zhu if you encounter any problems. 
