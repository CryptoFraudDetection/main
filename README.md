# main
Main repository for the project.

## Instructions

1. Clone the repo 📂
2. Install the project using the `make dev-install` command 🛠️
3. You're ready to start working ☕️
 
## Structure

    ├── .github/workflows         <- Github actions workflows.
    ├── data       
    │   ├── processed             <- The final, canonical data sets for modeling.
    │   └── raw                   <- The original, immutable data dump.
    │       
    ├── docs                      <- Documentation for the project.
    ├── models                    <- Modelcheckpoints, model predictions, metrics, and model summaries.
    ├── notebooks                 <- Jupyter notebooks or Quarto Markdown Notebooks. 
    │                                Naming convention is a number (for ordering) and a short `-` 
    │                                delimited description, e.g. `00-example.qmd`.
    │        
    ├── reports                   <- Generated analysis as HTML, PDF, LaTeX, diagrams, etc.
    ├── src/CryptoFraudDetection  <- Source code package for use in this project.
    ├── tests                     <- Unit tests for the project.
    ├── .gitignore                <- Files to be ignored by git.
    ├── compose.yml               <- Docker compose file for running the image
    ├── Dockerfile                <- Dockerfile for the Docker image.
    ├── LICENSE                   <- MIT License.
    ├── Makefile                  <- Makefile with commands like `make install` or `make test`.
    ├── pyproject.toml            <- Package build configuration.
    └── README.md                 <- The top-level README for this project.
