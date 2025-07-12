
# Define the step activation and perceptron
def step(x):
    return 1 if x >= 0 else 0

def perceptron(x1, x2, w1, w2, bias):
    return step(x1 * w1 + x2 * w2 + bias)

# Define logic gate functions
def not_gate(x):
    return step(-x + 0.5)

# Use perceptrons for gates
inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

# Evaluate all gates
results = {
    "AND": [perceptron(x[0], x[1], 1, 1, -1.5) for x in inputs],
    "OR": [perceptron(x[0], x[1], 1, 1, -0.5) for x in inputs],
    "NAND": [perceptron(x[0], x[1], -1, -1, 1.5) for x in inputs],
    "NOR": [perceptron(x[0], x[1], -1, -1, 0.5) for x in inputs],
    "NOT x1": [not_gate(x[0]) for x in inputs],
    "NOT x2": [not_gate(x[1]) for x in inputs],
    "XNOR (manually built from logic)": [  # using AND of NOR and AND
        perceptron(
            perceptron(x[0], x[1], -1, -1, 0.5),  # NOR(x1, x2)
            perceptron(x[0], x[1], 1, 1, -1.5),   # AND(x1, x2)
            1, 1, -1.5
        )
        for x in inputs
    ]
}

# Print the results
print("Logic Gate Truth Tables:")
print("-----------------------")
for gate, outputs in results.items():
    print(f"\n{gate} Gate:")
    print("  x1 | x2 | Output")
    print("------------------")
    for input, output in zip(inputs, outputs):
        print(f"   {input[0]}  |  {input[1]}  |   {output}")