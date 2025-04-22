import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split, SubsetRandomSampler
import matplotlib.pyplot as plt
import tqdm
import joblib

class Client:
    def __init__(self, id, data, model, lr=0.01, device=device):
        self.id = id
        self.data = data
        self.device = device
        self.model = model.to(device)
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=lr
        )
        self.loss_func = nn.MSELoss()

    def train(self, num_epochs, params, progress_bar):
        self.model.load_state_dict(params)
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch_idx, (x, y) in enumerate(self.data):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.model(x)
                loss = self.loss_func(y_pred, y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            progress_bar.update(1)
        progress_bar.set_postfix({"loss": running_loss / len(self.data)})
        return running_loss / len(self.data)

    def get_params(self):
        return self.model.state_dict()

    def set_params(self, params):
        self.model.load_state_dict(params)


class Server:
    def __init__(self, clients, model):
        self.clients = clients
        self.model = model
        self.loss_func = nn.MSELoss()

    def aggregate(self, sample_clients):
        params = [client.get_params() for client in sample_clients]
        avg_params = {}
        for key in params[0].keys():
            avg_params[key] = torch.stack(
                [params[i][key] for i in range(len(params))], 0
            ).mean(0)
        self.model.load_state_dict(avg_params)

    def test(self, data):
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(data):
                y_pred = self.model(x)
                loss = self.loss_func(y_pred, y)
                test_loss += loss.item()
        return test_loss / len(data)



    def get_params(self):
        return self.model.state_dict()

    def set_params(self, params):
        self.model.load_state_dict(params)

def fedSgdPar(
    model=LinearRegressionModel(),
    T=5,
    K=10,
    C=1,
    E=10,
    B=128,
    lr=0.01,
    verbose=True,
    noiid=False,
):
    if verbose:
        print("Running the Parallel implementation FedSGD on California Housing dataset")
        print(f"- Parameters: T={T}, K={K}, C={C}, E={E}, B={B}, lr={lr}, patience={patience}")
        print(f"- Model: {model.__class__.__name__}")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    clients_each_round = max(int(K * C), 1)
    california = fetch_california_housing()
    X = california.data[:, [california.feature_names.index("Population"), california.feature_names.index("HouseAge")]].reshape(-1, 2)  
    y = california.target.reshape(-1, 1)  

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    trainset = TensorDataset(X_train, y_train)
    testset = TensorDataset(X_test, y_test)

    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = random_split(trainset, [train_size, val_size])

    if verbose:
        print("- Data Split: ", len(trainset), len(valset), len(testset))

    trainloader = []
    if noiid:
        feature_idx = 0
        features = [trainset[i][0][feature_idx] for i in range(len(trainset))]
        sorted_indices = np.argsort(features)
        split_indices = np.array_split(sorted_indices, K)
        for i in range(K):
            sampler = SubsetRandomSampler(split_indices[i])
            loader = DataLoader(trainset, batch_size=B, sampler=sampler)
            trainloader.append(loader)
    else:
        indices = list(range(len(trainset)))
        random.shuffle(indices)
        split_indices = np.array_split(indices, K)
        for i in range(K):
            sampler = SubsetRandomSampler(split_indices[i])
            loader = DataLoader(trainset, batch_size=B, sampler=sampler)
            trainloader.append(loader)

    valoader = DataLoader(valset, batch_size=B, shuffle=True)
    testloader = DataLoader(testset, batch_size=B, shuffle=True)

    clients = []
    for i in range(K):
        client = Client(
            i,
            trainloader[i],
            LinearRegressionModel(),
            lr=lr,
            device=device,
        )
        clients.append(client)

    server = Server(clients, model)

    val_losses = []
    for r in range(T):
        params = server.get_params()
        progress_bar = tqdm.tqdm(
            total=E * clients_each_round, position=0, leave=False, desc="Round %d" % r
        )
        random_clients = random.sample(clients, clients_each_round)
        joblib.Parallel(n_jobs=10, backend="threading")(
            joblib.delayed(client.train)(E, params, progress_bar)
            for client in random_clients
        )
        server.aggregate(random_clients)
        val_loss = server.test(valoader)
        val_losses.append(val_loss)
        if verbose:
            print("Server - Val loss: %.3f" % (val_loss))

    test_loss = server.test(testloader)
    print("-- Test loss: %.3f --" % (test_loss))

    result = {
        "test_loss": test_loss,
        "val_losses": val_losses,
    }

    plot_model(model, testloader)


    return result


def plot_model(model, testloader):
    model.eval() 
    predictions = []
    true_values = []
    features = []

    with torch.no_grad():
        for X_batch, y_batch in testloader:
            y_pred = model(X_batch)
            predictions.extend(y_pred.numpy())
            true_values.extend(y_batch.numpy())
            features.extend(X_batch.numpy())

    predictions = np.array(predictions)
    true_values = np.array(true_values)
    features = np.array(features)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(features[:, 0], features[:, 1], true_values, c='b', marker='o', label='True Values')

    x = np.linspace(features[:, 0].min(), features[:, 0].max(), 10)
    y = np.linspace(features[:, 1].min(), features[:, 1].max(), 10)
    x, y = np.meshgrid(x, y)
    z = model.linear.weight[0, 0].item() * x + model.linear.weight[0, 1].item() * y + model.linear.bias.item()

    ax.plot_surface(x, y, z, alpha=0.5, color='r')

    ax.set_xlabel('Population')
    ax.set_ylabel('HouseAge')
    ax.set_zlabel('Target')
    ax.set_title('3D Scatter Plot of Features vs Target with Regression Plane')
    ax.legend()

    plt.show()

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(2, 1) 

    def forward(self, x):
        return self.linear(x)


fed_res = fedSgdPar(model=LinearRegressionModel(), T=40, K=100, C=0.1, E=5, B=10, lr=0.0001, noiid=True, verbose=False)
