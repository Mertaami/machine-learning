import numpy as np
import matplotlib.pyplot as plt

class LinearRegression():
    def __init__(self, input_dimension: int, learning_rate: float = 0.05):
        self.coef = np.zeros(input_dimension + 1)
        self.learning_rate = learning_rate

    def compute(self, input):
        result = self.coef @ np.insert(input, 0, 1)
        return result

    def learn(self, input_vec: np.ndarray, expected_vec: np.ndarray):
        input_vec = np.c_[np.ones(len(input_vec)), input_vec]
        self.coef = np.linalg.inv( input_vec.transpose() @ input_vec ) @ input_vec.transpose() @ expected_vec



if __name__ == "__main__":
    input_vec = np.array([
        [80, 27, 89],
        [80, 27, 88],
        [75, 25, 90],
        [62, 24, 87],
        [62, 22, 87],
        [62, 23, 87],
        [62, 24, 93],
        [62, 24, 93],
        [58, 23, 87],
        [58, 18, 80],
        [58, 18, 89],
        [58, 17, 88],
        [58, 18, 82],
        [58, 19, 93],
        [50, 18, 89],
        [50, 18, 86],
        [50, 19, 72],
        [50, 19, 79],
        [50, 20, 80],
        [56, 20, 82],
        [70, 20, 91]
    ])
    expected_vec = np.array([
        42,
        37,
        37,
        28,
        18,
        18,
        19,
        20,
        15,
        14,
        14,
        13,
        11,
        12,
        8,
        7,
        8,
        8,
        9,
        15,
        15
    ])
    errors = []
    assert len(expected_vec) == len(input_vec)
    print(f"Input shape: {input_vec[0].shape}")
    print(f"Initializating LinearRegression model...            ", end="")
    lin = LinearRegression(3, 1)
    print("OK")
    print(f"Testing compute...                                  ", end="")
    result = lin.compute(input_vec[0])
    assert result == 0.0
    print("OK")
    print("Testing learn...                                     ", end="")
    lin.learn(input_vec, expected_vec)
    print("OK")
    print("Testing compute...                                   ", end="")
    result = lin.compute(input_vec[0])
    print("OK")
    print(f"    {input_vec[0]} -> {result} ({expected_vec[0]})")
