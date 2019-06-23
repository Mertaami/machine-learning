# List of algorithms

* MultiLayerPerceptron
* Auto-encoder
* LinearRegression

## Linear Regression
`linear_regression.py`

```
class LinearRegression(input_dimension: int, learning_rate: float):
    compute(
        input: ndarray[input_dimension]
    ) -> output: float
    learn(
        input_vec: ndarray[n](ndarray[input_dimension]),
        expected_vec: ndarray[n](float)
    ) -> None
```
