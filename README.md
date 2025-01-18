# Main

Main repository for the project.

## Instructions

1. Clone the repo 📂
2. Install the project using the `make dev-install` command 🛠️
3. Copy the `.env-example` file to `.env` and fill in the necessary environment variables 🔑
4. Load the environment variables using the `source .env` command 🔄
5. You're ready to start working ☕️

## Structure

```
├── .github/workflows         <- Github actions workflows.
├── data       
│   ├── processed             <- The final, canonical data sets for modeling (parquet).
│   └── raw                   <- The original, immutable data dump.
│       
├── docs                      <- Documentation for the project. Papers, Docs and Lernplan for Challenge X (FHNW).
├── logs                      <- Logs for the project.
├── notebooks                 <- Jupyter or Quarto Markdown Notebooks.
│   │                            Naming convention is a number (for ordering) and a short `-`
│   │                            delimited description, e.g. `00-example.qmd`.
│   │                            
│   │                            OVERVIEW NOTEBOOKS:
│   │                            - 01 to 08:    Scraping and Data Collection
│   │                            - 09 to 14:    EDA and Data Preprocessing
│   │                            - 15:          run Pipeline
│   │                            - 18:          Multirocket main Model
│   │                            - 20:          run Embeddings
│   │                            
│   │                            not relevant for Challenge-X (FHNW):
│   │                            - 50 to 53:    Models of Gabriel Torres Gamez for cgml/5Da
│   │                            - 90:          old scraping approach for WDB
│   │
│   └── html_notebooks        <- relevant HTML versions of the notebooks. 
│    
├── scripts                   <- Scripts for the project.
├── src
│   ├── ConvTran              <- Source code package of external repository of pre-trained model. (subrepo) 
│   ├── CryptoFraudDetection  <- Source code package for use in this project.
│   └── CryptoFraudDetection  <- Package metadata.
│       .egg-info
│
├── tests                     <- Unit tests for the project. Includes scrapers, sentiment, embeddings, etc.
├── .env-template             <- Template for environment variables.
├── .gitignore                <- Files to be ignored by git.
├── compose.yml               <- Docker compose file for running the image.
├── Dockerfile                <- Dockerfile for the Docker image.
├── LICENSE                   <- MIT License.
├── Makefile                  <- Makefile with commands like `make install` or `make test`.
├── pyproject.toml            <- Package build configuration.
└── README.md                 <- The top-level README for this project.
```

## Train Models with Slurm

### Dummy Model Example

1. Clone the repo with a Personal Access Token (PAT) (use a classic token!):
   ```bash
   git clone https://USER:TOKEN@github.com/CryptoFraudDetection/main.git
   ```
   Replace `USER` with your GitHub username and `TOKEN` with your PAT.
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
   Follow the instructions on the terminal.
5. Initialize the sweep on your laptop or on Slurm:
   - Laptop:
     ```bash
     python scripts/dummy.py
     ```
   - Slurm:
     ```bash
     sbatch scripts/dummy.sh
     ```
6. Add agents to the sweep (if needed):
   1. Get the sweep ID from the log file from the previous step:
      - Slurm:
        ```bash
        cat logs/dummy*NNNN*.log
        ```
        Replace `NNNN` with the batch number.
      - Laptop: The sweep ID is printed on the terminal.
   2. Add agents to the sweep:
      ```bash
      sbatch scripts/dummy_sweep_agent.sh nod0ndel/dummy-model-sweep/_________
      ```
      Replace `_________` with the sweep ID.
7. List your jobs:
   ```bash
   squeue -u $USER
   ```
8. Check the logs:
   ```bash
   tail -f -n 100 logs/dummy*NNNN*.log
   ```

### Using Jupyter Notebooks on the Cluster

1. Run a Jupyter server in the current directory:
   ```bash
   cd ~/code/github.com/CryptoFraudDetection/main
   /cluster/common/jupyter/start-jupyter.sh -g 1 -c 12 -m 16384 -t 1-00:00:00 -d .
   ```
   - `-g`: Number of GPUs.
   - `-c`: Number of cores.
   - `-m`: Memory in MB.
   - `-t`: Runtime (e.g., `1-00:00:00` = 1 day).

2. Connect to the slave server using the command provided in the terminal output, e.g.,
   ```bash
   ssh -N -L 8888:localhost:8888 user@0.0.0.0
   ```
   - `-N`: No shell login.
   - `-L`: Traffic redirection.
   - `0.0.0.0`: The IP address of the slave server (provided in the terminal output).

3. Open the link provided in the terminal (e.g., `http://localhost:8888/lab?token=5ebfa321c439644dfa97c44fb96fc9e0296fec315ccc0f6f`) in your browser.

4. Install libraries in JupyterLab:
   ```bash
   !pip install -r requirements.txt
   ```
   Or install specific libraries:
   ```bash
   !pip install torch
   ```

### Tips and Best Practices

#### Slurm Job Management
- `srun`: Interactive job execution.
- `sbatch`: Batch job execution.
- `scancel`: Cancel a job.

#### Using Screen Sessions
- Start a `screen` session to prevent job interruption during network issues:
  ```bash
  screen
  ```
  Run commands inside the session.
- Detach the session: `Ctrl + A`, then `Ctrl + D`.
- Reconnect to the session:
  ```bash
  screen -rx
  ```

#### Batch Script Example

See the `scripts/dummy.sh` file for an example of a batch script.

### File Transfer to/from Slurm Server

- Transfer files to the Slurm server:
  ```bash
  scp -r . slurm:/path/to/remote/directory
  ```
- Transfer files from the Slurm server to local:
  ```bash
  scp -r slurm:/path/to/remote/file ./local/directory
  ```

