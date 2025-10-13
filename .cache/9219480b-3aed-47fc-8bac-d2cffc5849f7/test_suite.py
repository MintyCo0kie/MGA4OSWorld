import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from tetris import Tetris
from settings import BOARD_HEIGHT, BOARD_WIDTH


def test():
    game = Tetris(BOARD_HEIGHT, BOARD_WIDTH)
    game.new_block()
    while game.block.x > 0:
        game.move(-1, 0)
    game.rotate()
    if game.intersect():
        return False
    while game.block.x + len(game.block.shape[0]) < game.width:
        game.move(1, 0)
    game.rotate()
    if game.intersect():
        return False

    return True
