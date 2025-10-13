# snake.py
import pygame
from settings import *

class Snake:
    def __init__(self):
        self.length = 1
        self.positions = [((WIDTH // 2), (HEIGHT // 2))]
        self.direction = pygame.K_RIGHT
        self.color = GREEN

    def draw(self, surface):
        for pos in self.positions:
            rect = pygame.Rect((pos[0], pos[1]), (SNAKE_SIZE, SNAKE_SIZE))
            pygame.draw.rect(surface, self.color, rect)

    def move(self):
        cur = self.positions[0]
        x, y = cur
        if self.direction == pygame.K_UP:
            y -= SNAKE_SIZE
        elif self.direction == pygame.K_DOWN:
            y += SNAKE_SIZE
        elif self.direction == pygame.K_LEFT:
            x -= SNAKE_SIZE
        elif self.direction == pygame.K_RIGHT:
            x += SNAKE_SIZE
        new_head = (x, y)

        self.positions = [new_head] + self.positions[:-1]

    def grow(self):
        self.length += 1
        self.positions.append(self.positions[-1])