# food.py
import pygame
import random
from settings import *

class Food:
    def __init__(self):
        self.position = (random.randint(0, WIDTH-SNAKE_SIZE), random.randint(0, HEIGHT-SNAKE_SIZE))
        self.color = RED

    def draw(self, surface):
        rect = pygame.Rect((self.position[0], self.position[1]), (SNAKE_SIZE, SNAKE_SIZE))
        pygame.draw.rect(surface, self.color, rect)

    def respawn(self):
        self.position = (random.randint(0, (WIDTH - SNAKE_SIZE) // SNAKE_SIZE) * SNAKE_SIZE,
                         random.randint(0, (HEIGHT - SNAKE_SIZE) // SNAKE_SIZE) * SNAKE_SIZE)