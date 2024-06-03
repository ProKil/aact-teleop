# Teleop: VR- and websocket-based teleoperation

## Installation

### Python

Install poetry with either

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

or conda/mamba

```bash
conda install poetry
```

And then install the python project (Python version=3.11) with poetry

```bash
poetry install
```

### Environment variables

Set the ip address of the robot and Meta Quest device in `.env` file (Example: `.env.example`).

## Run the code

### Server

The server code has to be run on the Stretch Robot.

```bash
poetry run uvicorn teleop.server:app --host 0.0.0.0
```

### Client

The client code can be run on any device that could access the server,

```bash
python -m teleop.client
```

### Unity

The unity frontend need to be open with Unity Editor.


## Contribution

Please install pre-commit if you want to contribute to this project.

```bash
pip install pre-commit
pre-commit install
```
