import numpy as np
import random
import matplotlib.pyplot as plt
from multilayerperceptron import MultiLayerPerceptron

class AutoEncoder(MultiLayerPerceptron):
    def __init__(self, layer_sizes, features_layer_index):
        # Auto encoder has same input and output sizes
        assert layer_sizes[0] == layer_sizes[-1]
        return super().__init__(layer_sizes)
        self.features_layer_index = features_layer_index

    def __repr__(self):
        return "<AutoEncoder {}".format(self.layers)

    def encode(self, input) -> np.ndarray:
        for layer in self.layers[:self.features_layer_index+1]:
            input = layer.compute(input)
        return input

    def backprop(self, input):
        return super().backprop(input, input)


if __name__ == "__main__":
    # MNIST
    print("[Initializing] Perceptron: ", end="")
    #mlp = MultiLayerPerceptron([28*28, 32, 10, 32, 28*28])
    mlp = MultiLayerPerceptron.load('auto.npy')
    print("OK")
    print("[Loading] Training set: ", end="")
    space = np.load(open("data/train.npy", "rb"), allow_pickle=True)
    space = [(input.reshape(28*28), input.reshape(28*28)) for input, expected in space]
    print("OK    Testing set: ", end="")
    test_space = np.load(open("data/test.npy", "rb"), allow_pickle=True)
    test_space = [(input.reshape(28*28), input.reshape(28*28)) for input, expected in test_space]
    print("OK")

    mean_errors, mean_accuracy = mlp.learn(space, test_space, iterations=100, epochs=1000, save='auto.npy')

    image, expected = random.choice(space)
    input = np.reshape(image, 28*28)
    guess = mlp.frontprop(input)
    plt.figure(1)
    plt.subplot(211)
    plt.imshow(image.reshape((28, 28)), cmap='gray_r')
    plt.subplot(212)
    plt.imshow(guess.reshape((28,28)), cmap='gray_r')
    plt.show()
    # print(np.argmax(mlp.frontprop(input)), "=>", np.argmax(expected))

    plt.figure(1)
    #plt.subplot(211)
    plt.plot(mean_errors)
    plt.ylabel("error")
    #plt.subplot(212)
    #plt.plot(mean_accuracy)
    #plt.ylabel("accuracy")
    #plt.axis([0, len(mean_accuracy), 0, 1])
    plt.show()
