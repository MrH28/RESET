# RESET

A collection of simple Python graphics examples to help you get started with graphical programming.

---

## Prerequisites

- Python 3.7 or later — [download here](https://www.python.org/downloads/)

---

## Examples

| File | Library | What it shows |
|------|---------|---------------|
| `examples/matplotlib_example.py` | matplotlib | Sine wave plot |
| `examples/turtle_example.py` | turtle *(built-in)* | Colourful spiral |
| `examples/pygame_example.py` | pygame | Bouncing ball animation |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/MrH28/RESET.git
cd RESET
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install matplotlib pygame
```

> **Note:** The `turtle` example uses Python's built-in `turtle` module, so no extra installation is needed for that one.

---

## Running the examples

### Sine wave (matplotlib)

```bash
python examples/matplotlib_example.py
```

A window will open showing a sine wave plot.

### Colourful spiral (turtle)

```bash
python examples/turtle_example.py
```

A window will open and draw a colourful spiral.  Close the window when done.

### Bouncing ball (pygame)

```bash
python examples/pygame_example.py
```

A window will open with a bouncing ball animation.  Close the window or press the ✕ button to quit.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'matplotlib'` | Run `pip install matplotlib` |
| `ModuleNotFoundError: No module named 'pygame'` | Run `pip install pygame` |
| Turtle window closes immediately | Make sure you are running the script directly (`python examples/turtle_example.py`), not importing it |
| No display / headless server | For servers without a screen, install a virtual display: `sudo apt-get install xvfb` then prefix commands with `xvfb-run` |