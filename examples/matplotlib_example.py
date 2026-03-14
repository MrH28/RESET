"""
matplotlib_example.py
A simple example that draws a sine wave using matplotlib.

Requirements:
    pip install matplotlib

Run:
    python examples/matplotlib_example.py
"""

import math
import matplotlib.pyplot as plt

# Generate x values from 0 to 2π (64 steps of 0.1 covers 0 to 6.3 ≈ 2π)
x = [i * 0.1 for i in range(64)]
y = [math.sin(v) for v in x]

plt.figure(figsize=(8, 4))
plt.plot(x, y, color="steelblue", linewidth=2)
plt.title("Sine Wave")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.grid(True)
plt.tight_layout()
plt.show()
