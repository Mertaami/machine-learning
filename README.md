# List of algorithms

* [x] Linear Regression
* Logistic Regression
* Descision Tree
* [x] MultiLayer Perceptron
* Auto-encoder

## Linear Regression
`linear_regression.py`

```
class LinearRegression(input_dimension: int):
    compute(
        input: ndarray[input_dimension]
    ) -> output: float
    learn(
        input_vec: ndarray[n](ndarray[input_dimension]),
        expected_vec: ndarray[n](float)
    ) -> None
```
