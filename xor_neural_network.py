from typing import Any

import numpy as np
from numpy import ndarray, dtype, floating, complexfloating


# 활성화 함수와 그 도함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# 손실 함수
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


# 신경망 클래스
class NeuralNetwork:
    output: float | Any
    output_layer_activation: (
        float
        | ndarray[Any, dtype[floating[Any]]]
        | ndarray[Any, dtype[complexfloating[Any, Any]]]
        | Any
    )
    hidden_layer_output: float | Any
    hidden_layer_activation: (
        float
        | ndarray[Any, dtype[floating[Any]]]
        | ndarray[Any, dtype[complexfloating[Any, Any]]]
        | Any
    )

    def __init__(self, input_size, hidden_size, output_size):
        # Weight initialization
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)

        # Bias initialization
        self.bias_hidden = np.random.rand(1, hidden_size)
        self.bias_output = np.random.rand(1, output_size)

    def feedforward(self, X):
        # Hidden layer
        self.hidden_layer_activation = (
            np.dot(X, self.weights_input_hidden) + self.bias_hidden
        )
        self.hidden_layer_output = sigmoid(self.hidden_layer_activation)

        # Output layer
        self.output_layer_activation = (
            np.dot(self.hidden_layer_output, self.weights_hidden_output)
            + self.bias_output
        )
        self.output = sigmoid(self.output_layer_activation)

        return self.output

    def backpropagation(self, X, y, learning_rate):
        # Output layer error
        output_error = y - self.output
        output_delta = output_error * sigmoid_derivative(self.output)

        # Hidden layer error
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_layer_output)

        # Update weights and biases
        self.weights_hidden_output += (
            self.hidden_layer_output.T.dot(output_delta) * learning_rate
        )
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.feedforward(X)
            self.backpropagation(X, y, learning_rate)
            if epoch % 1000 == 0:
                loss = mse_loss(y, self.output)
                print(f"Epoch {epoch}, Loss: {loss}")


# XOR 문제 데이터셋
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

y = np.array([[0], [1], [1], [0]])

# 신경망 초기화
nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1)

# 훈련
nn.train(X, y, epochs=10000, learning_rate=0.1)

# 테스트
output = nn.feedforward(X)
print("Predicted Output:")
print(output)
