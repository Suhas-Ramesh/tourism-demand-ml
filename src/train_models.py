import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch


# Train sklearn models
def train_and_evaluate(model, X_train, y_train, X_test, y_test):

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return rmse, mae, r2


# Train neural network
def train_neural_network(model, X_train, y_train, X_test, y_test, epochs=100):

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)

    X_test_tensor = torch.FloatTensor(X_test)

    for epoch in range(epochs):

        model.train()

        optimizer.zero_grad()

        outputs = model(X_train_tensor)

        loss = criterion(outputs, y_train_tensor)

        loss.backward()

        optimizer.step()

    model.eval()

    with torch.no_grad():

        predictions = model(X_test_tensor).numpy().flatten()

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return rmse, mae, r2