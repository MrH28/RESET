"""
turtle_example.py
A simple example that draws a colourful spiral using Python's built-in
turtle module.  No extra packages are needed.

Run:
    python examples/turtle_example.py
"""

import turtle

colors = ["red", "orange", "yellow", "green", "blue", "purple"]

screen = turtle.Screen()
screen.bgcolor("black")
screen.title("Turtle Spiral")

pen = turtle.Turtle()
pen.speed(0)
pen.width(2)

for i in range(360):
    pen.color(colors[i % len(colors)])
    pen.forward(i * 0.5)
    pen.left(59)

pen.hideturtle()
screen.mainloop()
