import torch
import torch.nn as nn


class MyNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


CHECKPOINT_PATH = "C:/Users/DELL/OneDrive/Documents/GitHub/machine_failure/model/model.pt"

checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

model = MyNN(
    input_dim=checkpoint["input_dim"],
    hidden_dim=checkpoint["hidden_dim"],
    dropout_rate=checkpoint["dropout_rate"]
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def infer(features):
 
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        prob = model(x).item()
        pred = int(prob > 0.5)

    return {
        "prediction": pred,
        "probability": prob
    }


if __name__ == "__main__":
    input = [0.1] * checkpoint["input_dim"]
    result = infer(input)
    print(result)
