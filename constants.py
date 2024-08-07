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
NUM_MEDICINE = 0
NUM_WALLS = 5
SEED = 62    # Seed to determine random numbers
IS_RANDOM = True   # Set True for randomized layouts every episode, False for same layout
# MAP = None
MAP = [     # 1 = Wall 2 = Medicine 3 = Target 4 = Robot
      [4,0,0,0,0],
      [0,0,0,2,0],
      [0,1,0,0,0],
      [1,0,0,1,0],
      [1,0,0,1,3]
      ]

# Robot physics
BASE_FPS = 60
ROBOT_SPEED = 0.1
ROBOT_TURNING_SPEED = 0.1
FOV = 90   # <- enter in degrees
RAYS = 15    # number of distance rays casted

# Graphics
ROBOT_SPRITE = "sprites/robot_sprite.png"
RENDER_DURING_TRAINING = False

# Neural Network
HIDDEN_LAYERS = 256
MAX_STEPS = 200    # <- episode will be truncated after this number of steps