# main.py
import pygame
import sys
from settings import *
from snake import Snake
from food import Food

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    snake = Snake()
    food = Food()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    snake.direction = event.key

        snake.move()
        if snake.positions[0] == food.position:
            snake.grow()
            food.respawn()

        screen.fill(BLACK)
        snake.draw(screen)
        food.draw(screen)
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()