# code_structure

A short description of the project.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

Running the Pipeline

All commands should be run from the project root (MLOps/).

1. Preprocess the data

This loads the raw corrupted MNIST files, normalizes them, and saves processed tensors.

uv run python src/code_structure/data.py

2. Train the model

This trains a CNN on the processed dataset and saves:

the trained model to models/model.pth

training statistics to reports/figures/

uv run python src/code_structure/train.py


Optional hyperparameters:

uv run python src/code_structure/train.py --lr 1e-4 --batch-size 64 --epochs 20

3. Evaluate the model

Evaluate the trained model on the test set:

uv run python src/code_structure/evaluate.py models/model.pth

4. Visualize results

Generate visualizations (e.g. sample images or predictions):

uv run python src/code_structure/visualize.py


Plots are saved to:

reports/figures/
