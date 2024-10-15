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
        model = DummyModel(config.input_dim)
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        for epoch in range(config.epochs):
            outputs = model(torch.randn(64, config.input_dim))
            labels = torch.randn(64, 1)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            time.sleep(1)  # fake training time
            wandb.log({"epoch": epoch, "loss": loss.item()})
