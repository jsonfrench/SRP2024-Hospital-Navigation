# Pygame display
FPS = 60

# Hospital layout
GRID_ROWS = 5   # height
GRID_COLS = 5  # width
CELL_HEIGHT = 64 
CELL_WIDTH = 64 
# Hospital Generation
TOLERANCE = 10  # Number of times we check for a valid position to place a tile before deleting existing ones
NUM_TARGETS = 1 # <- currently limited to one target, a higher value will do nothing
NUM_MEDICINE = 1
NUM_WALLS = 10

# Robot physics
BASE_FPS = 60
ROBOT_SPEED = 0.1
ROBOT_TURNING_SPEED = 0.1

# Graphics
ROBOT_SPRITE = "sprites/robot_sprite.png"