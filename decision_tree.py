import numpy as np

class DecisionTree():
    def __init__(self, input_dimension: int):
        pass
    
    def learn(self, input_vec: np.ndarray, expected_vec: np/ndarray):
        
    


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
    assert len(input_vec) == len(expected_vec)
    print(f"Input shape: {input_vec[0].shape}")
    print(f"Initializating DecisionTree model...            ", end="")
    lin = DecisionTree(3)
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

