# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import time


# Dummy model
class DummyModel(nn.Module):
    def __init__(self, input_dim):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)


# Training function
def train_model(config=None):
    with wandb.init(config=config):
        config = wandb.config

        # Check for GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model = DummyModel(config.input_dim).to(device)
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        for epoch in range(config.epochs):
            inputs = torch.randn(64, config.input_dim).to(device)
            labels = torch.randn(64, 1).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            time.sleep(1)  # Fake training time
            wandb.log({"epoch": epoch, "loss": loss.item()})
