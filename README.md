# Main

Main repository for the project.

## Instructions

1. Clone the repo 📂
2. Install the project using the `make dev-install` command 🛠️
3. Copy the `.env-example` file to `.env` and fill in the necessary environment variables 🔑
4. Load the environment variables using the `source .env` command 🔄
5. You're ready to start working ☕️

## Installing `uv`

`uv` is a universal runtime tool for running and managing Python applications. It ensures a streamlined setup and cross-platform compatibility.

### macOS/Linux  
Run the following command in your terminal:  
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows  
Execute the following in PowerShell:  
```powershell
powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
```

## Setting Up the Environment

Run the command below to create a virtual environment and install all required dependencies:  
```bash
make install
```

### Notes:
- Ensure you have **`make`** installed on your system for the setup command.  
- For Windows users, you may need to install `make` using tools like **`choco`** or **`winget`**.  

## Structure

```
├── .github/workflows         <- Github actions workflows.
├── data       
│   ├── processed             <- The final, canonical data sets for modeling.
│   └── raw                   <- The original, immutable data dump.
│       
├── docs                      <- Documentation for the project.
├── models                    <- Model checkpoints, predictions, metrics, and summaries.
├── notebooks                 <- Jupyter or Quarto Markdown Notebooks.
│                                Naming convention is a number (for ordering) and a short `-`
│                                delimited description, e.g. `00-example.qmd`.
│        
├── reports                   <- Generated analysis as HTML, PDF, LaTeX, diagrams, etc.
├── scripts                   <- Scripts for data processing, model training, etc.
├── src/CryptoFraudDetection  <- Source code package for use in this project.
├── tests                     <- Unit tests for the project.
├── .gitignore                <- Files to be ignored by git.
├── compose.yml               <- Docker compose file for running the image.
├── Dockerfile                <- Dockerfile for the Docker image.
├── LICENSE                   <- MIT License.
├── Makefile                  <- Makefile with commands like `make install` or `make test`.
├── pyproject.toml            <- Package build configuration.
└── README.md                 <- The top-level README for this project.
```