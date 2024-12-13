# main
Main repository for the project.

## Instructions

1. Clone the repo 📂
2. Install the project using the `make dev-install` command 🛠️
3. Copy the `.env-example` file to `.env` and fill in the necessary environment variables 🔑
4. Load the environment variables using the `source .env` command 🔄
5. You're ready to start working ☕️
 
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
    ├── scripts                   <- Scripts for data processing, model training, etc.
    ├── src/CryptoFraudDetection  <- Source code package for use in this project.
    ├── tests                     <- Unit tests for the project.
    ├── .gitignore                <- Files to be ignored by git.
    ├── compose.yml               <- Docker compose file for running the image
    ├── Dockerfile                <- Dockerfile for the Docker image.
    ├── LICENSE                   <- MIT License.
    ├── Makefile                  <- Makefile with commands like `make install` or `make test`.
    ├── pyproject.toml            <- Package build configuration.
    └── README.md                 <- The top-level README for this project.

## Train models with Slurm

### Dummy Model Example

1. Clone the repo with a Personal Access Token (PAT) (use a classic token!):
    ```bash
    git clone https://USER:TOKEN@github.com/CryptoFraudDetection/main.git
    ```
    (Replace `USER` with your GitHub username and `TOKEN` with your PAT)
    ```bash
    cd main
    ```
2. Create a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3. Install the project:
    ```bash
    pip install -e .
    ```
4. Login to wandb:
    ```bash
    wandb login
    ```
    (Follow the instructions on the terminal)
3. Initialize the sweep on your laptop or on Slurm:
    - Laptop: 
        ```bash
        python scripts/dummy.py
        ```
    - Slurm:
        ```bash
        sbatch scripts/dummy.sh
        ```
4. Add agents to the sweep (if needed):
    1. Get the sweep id from the log file from the previous step:
        1. slurm:
            ```bash
            cat logs/dummy*NNNN*.log
            ```
            (Replace `NNNN` with the batch number)
        2. laptop: sweep id is printed on the terminal
    2. Add agents to the sweep:
        ```bash
        sbatch scripts/dummy_sweep_agent.sh nod0ndel/dummy-model-sweep/_________
        ```
        (Replace `_________` with the sweep id)
5. List your jobs:
    ```bash
    squeue -u $USER
    ```
6. Check the logs:
    ```bash
    tail -f -n 100 logs/dummy*NNNN*.log
    ```

1. Run Jupyter notebook on the cluster:

    Run a jupyter server in the current directory with the following command:

    ```bash
    cd ~/code/github.com/CryptoFraudDetection/main
    /cluster/common/jupyter/start-jupyter.sh -g 1 -c 12 -m 16384 -t 1-00:00:00 -d .
    ```
2. Connect to slave server with command on output
3. Install packages in jupyterlab:
    ```bash
    !pip install -r requirements.txt
    ```
    or
    ```bash
    !pip install torch
    ```
