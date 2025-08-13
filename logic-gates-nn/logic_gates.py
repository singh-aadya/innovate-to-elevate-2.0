import torch
import torch.nn as nn
import torch.optim as optim

# Define truth tables for each gate
truth_tables = {
    "AND":  (torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]),
             torch.tensor([[0.],[0.],[0.],[1.]])),
    "OR":   (torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]),
             torch.tensor([[0.],[1.],[1.],[1.]])),
    "XOR":  (torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]),
             torch.tensor([[0.],[1.],[1.],[0.]])),
    "NAND": (torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]),
             torch.tensor([[1.],[1.],[1.],[0.]])),
    "NOR":  (torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]),
             torch.tensor([[1.],[0.],[0.],[0.]])),
    "XNOR": (torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]),
             torch.tensor([[1.],[0.],[0.],[1.]])),
    "NOT":  (torch.tensor([[0.],[1.]]),
             torch.tensor([[1.],[0.]])),
}

# Small neural net class
class LogicGateNN(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 4),
            nn.Sigmoid(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Training loop for each gate
for gate_name, (X, y) in truth_tables.items():
    model = LogicGateNN(X.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    for epoch in range(1000):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = model(X).round()
        print(f"{gate_name} predictions: {preds.squeeze().tolist()}")
