'''
This module models the problem to be solved. In this very simple example, the problem is to optimze a Robot that works in a Warehouse.
The Warehouse is divided into a rectangular grid. A Target is randomly placed on the grid and the Robot's goal is to reach the Target.
'''
import random
from enum import Enum
import pygame
import sys
from os import path
import math
import time
import constants as const

# Actions the Robot is capable of performing i.e. go in a certain direction
class RobotAction(Enum):
    LEFT=0
    FORWARD=1
    RIGHT=2
    BACKWARD=3
    INTERACT=4

class WarehouseRobot:

    # Initialize the grid size. Pass in an integer seed to make randomness (Targets) repeatable.
    def __init__(self, grid_rows=const.GRID_ROWS, grid_cols=const.GRID_COLS, fps=const.FPS):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.reset()

        self.fps = fps
        self.last_action=''
        self._init_pygame()

    def _init_pygame(self):
        pygame.init() # initialize pygame
        pygame.display.init() # Initialize the display module

        # Game clock
        self.clock = pygame.time.Clock()

        # Default font
        self.action_font = pygame.font.SysFont("Calibre",30)
        self.action_info_height = self.action_font.get_height()

        # For rendering
        self.cell_height = const.CELL_HEIGHT
        self.cell_width = const.CELL_WIDTH
        self.cell_size = (self.cell_width, self.cell_height) 
        self.grid_line_width = int(min(self.cell_width, self.cell_height) * 0.07815)

        # For controlling robot's speed
        delta_time = const.BASE_FPS/self.fps    # Standardize movement to how it acts in 60fps. Breaks collision for low framerates.
        self.robot_speed = const.ROBOT_SPEED * delta_time
        self.robot_turning_speed = const.ROBOT_TURNING_SPEED * delta_time

        # Define game window size (width, height)
        self.window_size = (self.cell_size[0] * self.grid_cols, self.cell_size[1] * self.grid_rows + self.action_info_height)

        # Initialize game window
        self.window_surface = pygame.display.set_mode(self.window_size) 

        # Initialize and resize sprites
        file_name = path.join(path.dirname(__file__), const.ROBOT_SPRITE)
        img = pygame.image.load(file_name)
        self.robot_img = pygame.transform.scale(img, self.cell_size)

    def reset(self, seed=None):
        # Initialize Robot's starting position and attributes
        self.robot_pos = [
                random.randint(0, self.grid_cols-1),
                random.randint(0, self.grid_rows-1)
            ]
        self.robot_facing_angle = random.uniform(0.0, math.pi*2)
        self.medicine_amt = 0   # <- hard coded constant because robot should never start with more than 0 medicine

        # For generating map elements
        self.tolerance = const.TOLERANCE
        
        # Define number of tiles
        self.num_targets = const.NUM_TARGETS
        self.num_medicine = const.NUM_MEDICINE
        self.num_walls = const.NUM_WALLS

        if not self.num_medicine:
            self.medicine_amt = 1   # Start the robot with medicine to deliver if none are generated

        if self.num_targets + self.num_medicine + self.num_walls + 1 > self.grid_rows*self.grid_cols:   # Sum all tiles plus an extra space for the player
            raise ValueError(f"Number of tiles ({self.num_targets + self.num_medicine + self.num_walls + 1}) exceeds number of grid spaces ({self.grid_rows*self.grid_cols})")

        # Set random target position
        random.seed(seed)
        placements_left = self.num_targets
        while placements_left > 0:
            potential_pos = [
                random.randint(0, self.grid_cols-1),
                random.randint(0, self.grid_rows-1)
            ]
            if potential_pos == self.robot_pos:
                continue
            self.target_pos = potential_pos
            placements_left -= 1

        # Generate wall positons
        random.seed(seed)
        tolerance = self.tolerance
        self.wall_pos = []
        placements_left = self.num_walls
        while placements_left > 0:
            potential_pos = [
                random.randint(0, self.grid_cols-1),
                random.randint(0, self.grid_rows-1)
            ]
            if tolerance < 0:
                self.wall_pos.remove(self.wall_pos[random.randint(0,len(self.wall_pos)-1)])
                tolerance = self.tolerance
            if potential_pos == self.robot_pos:
                continue
            if potential_pos in self.wall_pos:
                continue
            if potential_pos == self.target_pos:
                continue
            self.wall_pos.append(potential_pos)
            if not self.dfs(self.robot_pos, self.target_pos):
                self.wall_pos.remove(potential_pos)
                tolerance -= 1
                continue
            placements_left -= 1

        # Generate medicine positons
        random.seed(seed)
        tolerance = self.tolerance  # How many times we attempt to place medicine before knocking down a wall
        self.medicine_pos = []
        placements_left = self.num_medicine
        while placements_left > 0:
            potential_pos = [
                random.randint(0, self.grid_cols-1),
                random.randint(0, self.grid_rows-1)
            ]
            if tolerance < 0:
                self.wall_pos.remove(self.wall_pos[random.randint(0,len(self.wall_pos)-1)])
                tolerance = self.tolerance
            if potential_pos == self.robot_pos:
                continue
            if potential_pos in self.medicine_pos:
                continue
            if potential_pos in self.wall_pos:
                continue
            if potential_pos == self.target_pos:
                continue
            if not self.dfs(potential_pos, self.target_pos):
                tolerance -= 1
                continue
            self.medicine_pos.append(potential_pos)
            placements_left -= 1

    def is_valid_player_pos(self, x, y, dx, dy, max_x, max_y, walls):
        valid_x = (
            0 < dx < max_x and
            [int(dx+.5),int(y+.5)] not in walls
        )
        valid_y = (
            0 < dy < max_y and
            [int(x+.5),int(dy+.5)] not in walls
        )
        return (valid_x, valid_y)
    
    # DFS to check for a valid path
    def dfs(self, position, destination):
        discovered = []
        frontier = []
        frontier.append(position)
        while frontier:
            position = frontier[len(frontier)-1]
            frontier.pop()
            if position not in discovered:
                discovered.append(position)
                directions = [(1,0),(0,1),(-1,0),(0,-1)]
                for x, y in directions:
                    x_new = position[0] + x
                    y_new = position[1] + y
                    if x_new < 0 or x_new >= self.grid_cols or y_new < 0 or y_new >= self.grid_rows:
                        continue
                    if [x_new, y_new] == destination:
                        return True
                    if [x_new, y_new] not in self.wall_pos:
                        frontier.append([x_new,y_new])
        return False

    # visulize DFS
    def vis_dfs(self, position, destination):
        discovered = []
        frontier = []
        frontier.append(position)
        while frontier:
            position = frontier[len(frontier)-1]
            frontier.pop()
            if position not in discovered:
                discovered.append(position)
                directions = [(1,0),(0,1),(-1,0),(0,-1)]
                for x, y in directions:
                    x_new = position[0] + x
                    y_new = position[1] + y
                    if x_new < 0 or x_new >= self.grid_cols or y_new < 0 or y_new >= self.grid_rows:
                        continue
                    if [x_new, y_new] == destination:
                        return (discovered, frontier)
                    if [x_new, y_new] not in self.wall_pos:
                        frontier.append([x_new,y_new])
        return (discovered, frontier)

    def perform_action(self, robot_action:RobotAction) -> bool:
        self.last_action = robot_action

        # Rotate left
        if robot_action == RobotAction.LEFT:
            self.robot_facing_angle -= self.robot_turning_speed
        # Rotate right
        elif robot_action == RobotAction.RIGHT:
            self.robot_facing_angle += self.robot_turning_speed
        # Move forward
        elif robot_action == RobotAction.FORWARD:
            desired_x = self.robot_pos[0] + math.cos(self.robot_facing_angle)*self.robot_speed
            desired_y = self.robot_pos[1] + math.sin(self.robot_facing_angle)*self.robot_speed
            if self.is_valid_player_pos(self.robot_pos[0], self.robot_pos[1], desired_x, desired_y, self.grid_cols-1, self.grid_rows-1, self.wall_pos)[0]:
                self.robot_pos[0] = desired_x
            if self.is_valid_player_pos(self.robot_pos[0], self.robot_pos[1], desired_x, desired_y, self.grid_cols-1, self.grid_rows-1, self.wall_pos)[1]:
                self.robot_pos[1] = desired_y
        # Move backward
        elif robot_action == RobotAction.BACKWARD:
            desired_x = self.robot_pos[0] - math.cos(self.robot_facing_angle)*self.robot_speed
            desired_y = self.robot_pos[1] - math.sin(self.robot_facing_angle)*self.robot_speed
            if self.is_valid_player_pos(self.robot_pos[0], self.robot_pos[1], desired_x, desired_y, self.grid_cols-1, self.grid_rows-1, self.wall_pos)[0]:
                self.robot_pos[0] = desired_x
            if self.is_valid_player_pos(self.robot_pos[0], self.robot_pos[1], desired_x, desired_y, self.grid_cols-1, self.grid_rows-1, self.wall_pos)[1]:
                self.robot_pos[1] = desired_y
        elif robot_action == RobotAction.INTERACT:
            robot_grid_pos = [int(self.robot_pos[0]+.5), int(self.robot_pos[1]+.5)]
            if robot_grid_pos in self.medicine_pos:
                self.medicine_amt += 1
                self.medicine_pos.remove(robot_grid_pos)
            elif robot_grid_pos == self.target_pos:
                self.medicine_amt = 0       # Do something else (to be implemented later)
        # Clamp facing angle to 0 - 6.2831
        self.robot_facing_angle %= math.pi*2

        return (self.medicine_pos, self.medicine_amt)

    def render(self):

        self._process_events()

        # Wipe the screen to get rid of any artifacts from last frame
        self.window_surface.fill((255,255,255))

        #Draw better background
        colors = [      # Requires at least two colors (set both to same color if you don't want a gradient)
            (255,255,255), 
            (200,200,200) 
        ]
        iterations = ((self.window_size[1] - self.action_info_height) // self.grid_line_width + 1) // (len(colors)-1) + 1
        for i in range(len(colors) - 1):
            for j in range(iterations):
                pygame.draw.line(self.window_surface, 
                                (colors[i][0]+ ((colors[i+1][0]-colors[i][0]) / iterations * j), 
                                 colors[i][1]+ ((colors[i+1][1]-colors[i][1]) / iterations * j), 
                                 colors[i][2]+ ((colors[i+1][2]-colors[i][2]) / iterations * j)), 
                                (0, j*self.grid_line_width + i*((self.window_size[1]-self.action_info_height) / (len(colors)-1))), 
                                (self.window_size[0], j*self.grid_line_width + i*((self.window_size[1]-self.action_info_height) / (len(colors)-1))), 
                                self.grid_line_width)

        # Draw target
        pygame.draw.rect(self.window_surface, "blue", pygame.Rect(self.target_pos[0]*self.cell_size[0], self.target_pos[1]*self.cell_size[1], self.cell_size[0], self.cell_size[1]))
        pygame.draw.rect(self.window_surface, (150,150,255), pygame.Rect(self.target_pos[0]*self.cell_size[0]+self.grid_line_width, self.target_pos[1]*self.cell_size[1]+self.grid_line_width, self.cell_size[0]-self.grid_line_width*2, self.cell_size[1]-self.grid_line_width*2))


        # Draw medicine
        for medicine in (self.medicine_pos):
            pygame.draw.rect(self.window_surface, (0,150,0), pygame.Rect(medicine[0]*self.cell_size[0], medicine[1]*self.cell_size[1], self.cell_size[0], self.cell_size[1]))
            pygame.draw.circle(self.window_surface, (150,255,150), (medicine[0]*self.cell_size[0]+self.cell_size[0]/2,medicine[1]*self.cell_size[1]+self.cell_size[1]/2), (min(self.cell_size[0], self.cell_size[1]))/4)

        # Draw walls
        for wall in (self.wall_pos):
            pygame.draw.rect(self.window_surface, "black", pygame.Rect(wall[0]*self.cell_size[0], wall[1]*self.cell_size[1], self.cell_size[0], self.cell_size[1]))

        # Draw pathfinding (for showcasing purposes -- not recommended on higher framerates)
        # for square in self.vis_dfs(self.robot_pos, self.target_pos)[1]:
        #     pygame.draw.rect(self.window_surface, (255,100,100), pygame.Rect(square[0]*self.cell_size[0], square[1]*self.cell_size[1], self.cell_size[0], self.cell_size[1]))
        # for square in self.vis_dfs(self.robot_pos, self.target_pos)[0]:
        #     pygame.draw.rect(self.window_surface, (255,75,75), pygame.Rect(square[0]*self.cell_size[0], square[1]*self.cell_size[1], self.cell_size[0], self.cell_size[1]))

        # Draw grid dots
        for i in range((self.window_size[1] - self.action_info_height) // self.cell_size[1] + 1):
            for j in range(self.window_size[0] // self.cell_size[0] + 1): 
                pygame.draw.circle(self.window_surface, "grey", (j*self.cell_size[0], i*self.cell_size[1]), self.grid_line_width / 2 )

        # Draw player
        player_center = (self.robot_pos[0]*self.cell_size[0]+self.cell_size[0]/2,self.robot_pos[1]*self.cell_size[1]+self.cell_size[1]/2)
        pygame.draw.circle(self.window_surface, "red", player_center, min(self.cell_size[0], self.cell_size[1])/2)
        pygame.draw.line(self.window_surface, "blue", player_center, (player_center[0] + math.cos(self.robot_facing_angle)*self.cell_size[0], player_center[1] + math.sin(self.robot_facing_angle)*self.cell_size[1]))

        # Draw action display
        pygame.draw.rect(self.window_surface, pygame.Color(colors[len(colors)-1]), pygame.Rect(0,self.window_size[1]-self.action_info_height, self.window_size[0], self.action_info_height))
        text_img = self.action_font.render(f'Action: {self.last_action}', True, (0,0,0), pygame.Color(colors[len(colors)-1]))
        text_pos = (0, self.window_size[1] - self.action_info_height)
        self.window_surface.blit(text_img, text_pos) 

        pygame.display.update()
                
        # Limit frames per second
        self.clock.tick(self.fps)  

    def _process_events(self):
        # Process user events, key presses
        for event in pygame.event.get():
            # User clicked on X at the top right corner of window
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if(event.type == pygame.KEYDOWN):
                # User hit escape
                if(event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    sys.exit()

# For unit testing
if __name__=="__main__":
    warehouseRobot = WarehouseRobot()
    warehouseRobot.render()

    while(True):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Manually run using keyboard
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]:
            warehouseRobot.perform_action(list(RobotAction)[0])
        if keys[pygame.K_w]:
            warehouseRobot.perform_action(list(RobotAction)[1])
        if keys[pygame.K_d]:
            warehouseRobot.perform_action(list(RobotAction)[2])
        if keys[pygame.K_s]:
            warehouseRobot.perform_action(list(RobotAction)[3])
        if keys[pygame.K_SPACE]:
            warehouseRobot.perform_action(list(RobotAction)[4])

        warehouseRobot.render()