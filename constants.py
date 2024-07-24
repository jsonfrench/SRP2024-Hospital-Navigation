# Pygame display
FPS = 60

# Hospital layout
GRID_ROWS = 4   # height
GRID_COLS = 4  # width
CELL_HEIGHT = 64 
CELL_WIDTH = 64 
# Hospital Generation
TOLERANCE = 10  # Number of times we check for a valid position to place a tile before deleting existing ones
NUM_TARGETS = 1 # <- currently limited to one target, a higher value will do nothing
NUM_MEDICINE = 1
NUM_WALLS = 4
SEED = 5    # Enter None for random, any integer for persistent environment 

# Robot physics
BASE_FPS = 60
ROBOT_SPEED = 0.1
ROBOT_TURNING_SPEED = 0.1

# Graphics
ROBOT_SPRITE = "sprites/robot_sprite.png"

# Neural Network
HIDDEN_LAYERS = 256
MAX_STEPS = 100000    # <- episdoe will be truncated after this number of steps