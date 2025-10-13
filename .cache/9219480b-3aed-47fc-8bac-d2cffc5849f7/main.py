# main.py
import pygame
import sys
from tetris import Tetris
import settings  # Import settings

# Initialize Pygame
pygame.init()

# Screen dimensions
screen = pygame.display.set_mode((settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT))
pygame.display.set_caption("Tetris")

# Game setup
game = Tetris(settings.BOARD_HEIGHT, settings.BOARD_WIDTH)  # Use settings for board size
clock = pygame.time.Clock()
last_fall_time = pygame.time.get_ticks()

fall_time = 0
fall_speed = 100

# Main game loop
running = True
while running:
    clock.tick(settings.GAME_SPEED)

    if game.block is None:
        game.new_block()

    current_time = pygame.time.get_ticks()
    if current_time - last_fall_time > fall_speed:
        last_fall_time = current_time
        game.go_down()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                game.move(-1, 0)
            if event.key == pygame.K_RIGHT:
                game.move(1, 0)
            if event.key == pygame.K_DOWN:
                game.go_down()
            if event.key == pygame.K_UP:
                game.rotate()
            if event.key == pygame.K_SPACE:
                game.go_space()

    screen.fill(settings.BLACK)

    # Draw the game board
    for i in range(game.height):
        for j in range(game.width):
            pygame.draw.rect(screen, settings.GREY,
                             [settings.BLOCK_SIZE * j, settings.BLOCK_SIZE * i, settings.BLOCK_SIZE,
                              settings.BLOCK_SIZE], 1)
            if game.board[i][j] != 0:
                pygame.draw.rect(screen, settings.WHITE,
                                 [settings.BLOCK_SIZE * j + 1, settings.BLOCK_SIZE * i + 1, settings.BLOCK_SIZE - 2,
                                  settings.BLOCK_SIZE - 1])

    # Draw the current block
    if game.block is not None:
        for i in range(len(game.block.shape)):
            for j in range(len(game.block.shape[i])):
                if game.block.shape[i][j] != 0:
                    pygame.draw.rect(screen, game.block.color, [settings.BLOCK_SIZE * (j + game.block.x) + 1,
                                                                settings.BLOCK_SIZE * (i + game.block.y) + 1,
                                                                settings.BLOCK_SIZE - 2, settings.BLOCK_SIZE - 2])

    pygame.display.flip()
    clock.tick(settings.GAME_SPEED)

pygame.quit()
sys.exit()
