"""
pygame_example.py
A simple bouncing ball animation using pygame.

Requirements:
    pip install pygame

Run:
    python examples/pygame_example.py
"""

import sys
import pygame

# Window dimensions
WIDTH, HEIGHT = 600, 400
FPS = 60

# Ball properties
BALL_RADIUS = 20
BALL_COLOR = (70, 130, 180)   # steel blue
BG_COLOR = (30, 30, 30)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Bouncing Ball")
    clock = pygame.time.Clock()

    # Ball position and velocity
    x, y = WIDTH // 2, HEIGHT // 2
    vx, vy = 4, 3

    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Move ball
        x += vx
        y += vy

        # Bounce off walls
        if x - BALL_RADIUS < 0 or x + BALL_RADIUS > WIDTH:
            vx = -vx
        if y - BALL_RADIUS < 0 or y + BALL_RADIUS > HEIGHT:
            vy = -vy

        # Draw
        screen.fill(BG_COLOR)
        pygame.draw.circle(screen, BALL_COLOR, (int(x), int(y)), BALL_RADIUS)
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
