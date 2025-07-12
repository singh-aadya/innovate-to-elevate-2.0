def perceptron(x1, x2, w1, w2, bias):
    """
    Implements a simple perceptron with two inputs, two weights, and a bias.
    Returns 1 if the weighted sum plus bias is greater than 0, else returns 0.
    """
    weighted_sum = x1 * w1 + x2 * w2 + bias
    return 1 if weighted_sum > 0 else 0

def XOR_nn(x1, x2):
    # First layer (hidden layer)
    out1 = perceptron(x1, x2, 1, 1, -0.5)   # OR gate
    out2 = perceptron(x1, x2, -1, -1, 1.5)  # NAND gate

    # Second layer (output)
    return perceptron(out1, out2, 1, 1, -1.5)  # AND gate

print("XOR Gate (using multi-layer perceptron):")
for x in [(0,0), (0,1), (1,0), (1,1)]:
    print(f"{x} -> {XOR_nn(x[0], x[1])}")
