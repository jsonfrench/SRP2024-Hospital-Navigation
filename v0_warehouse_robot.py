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

class WarehouseRobot:

    # Initialize the grid size. Pass in an integer seed to make randomness (Targets) repeatable.
    def __init__(self, grid_rows=const.GRID_ROWS, grid_cols=const.GRID_COLS, fps=const.FPS, render_mode='human'):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.fps = fps
        self.render_mode = render_mode

        # For controlling robot's speed
        delta_time = const.BASE_FPS/self.fps    # Standardize movement to how it acts in 60fps. Breaks collision for low framerates.
        self.robot_speed = const.ROBOT_SPEED * delta_time
        self.robot_turning_speed = const.ROBOT_TURNING_SPEED * delta_time

        # For controlling robot's speed
        delta_time = const.BASE_FPS/self.fps    # Standardize movement to how it acts in 60fps. Breaks collision for low framerates.
        self.robot_speed = const.ROBOT_SPEED * delta_time
        self.robot_turning_speed = const.ROBOT_TURNING_SPEED * delta_time

        # For controlling robot's speed
        delta_time = const.BASE_FPS/self.fps    # Standardize movement to how it acts in 60fps. Breaks collision for low framerates.
        self.robot_speed = const.ROBOT_SPEED * delta_time
        self.robot_turning_speed = const.ROBOT_TURNING_SPEED * delta_time
        self.generate_hospital(None if const.IS_RANDOM else const.SEED)
        self.reset()
        self.last_action=''

        self.window_surface = None

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

        # Define game window size (width, height)
        self.window_size = (self.cell_size[0] * self.grid_cols, self.cell_size[1] * self.grid_rows + self.action_info_height)

        # Initialize and resize sprites
        file_name = path.join(path.dirname(__file__), const.ROBOT_SPRITE)
        img = pygame.image.load(file_name)
        self.robot_img = pygame.transform.scale(img, self.cell_size)

    def generate_hospital(self, seed=None):
        random.seed(seed)

        # For generating map elements
        self.tolerance = const.TOLERANCE
        
        # Define number of tiles
        self.num_targets = const.NUM_TARGETS
        self.num_medicine = const.NUM_MEDICINE
        self.num_walls = const.NUM_WALLS

        # Check to make sure all tiles can fit on the grid
        if self.num_targets + self.num_medicine + self.num_walls + 1 > self.grid_rows*self.grid_cols:   # Sum all tiles plus an extra space for the player
            raise ValueError(f"Number of tiles ({self.num_targets + self.num_medicine + self.num_walls + 1}) exceeds number of grid spaces ({self.grid_rows*self.grid_cols})")

        # Initialize Robot's starting position and attributes
        self.robot_pos = [
                random.randint(0, self.grid_cols-1),
                random.randint(0, self.grid_rows-1)
            ]
        self.robot_grid_pos = [int(self.robot_pos[0]+0.5),int(self.robot_pos[1]+0.5)]
        self.robot_facing_angle = random.uniform(0.0, math.pi*2)
        self.reset_robot_pos = self.robot_pos.copy()
        self.reset_robot_facing_angle = self.robot_facing_angle

        # Set random target position
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
        self.medicine_amt = 1 if not self.medicine_pos else 0
        self.reset_medicine_amt = self.medicine_amt
        self.reset_medicine_pos = self.medicine_pos.copy()

    def reset(self):
        if const.IS_RANDOM:
            self.generate_hospital(seed=None)
        self.robot_pos = self.reset_robot_pos.copy()
        self.robot_facing_angle = self.reset_robot_facing_angle
        self.robot_delta_pos = [math.cos(self.robot_facing_angle)*self.robot_speed,math.sin(self.robot_facing_angle)*self.robot_speed]
        self.medicine_pos = self.reset_medicine_pos.copy()
        self.medicine_amt = self.reset_medicine_amt

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
    
    def distance(self, ax, ay, bx, by):
        return math.sqrt((bx-ax)*(bx-ax)+(by-ay)*(by-ay))
    
    def raycast(self):
        fov = 60 * 0.0174533
        rays = 50
        ray_angle = self.robot_facing_angle - fov/2 if rays > 1 else self.robot_facing_angle
        px = self.robot_pos[0]*self.cell_size[0]+self.cell_size[0]/2    # player center screen coordinate x
        py = self.robot_pos[1]*self.cell_size[1]+self.cell_size[1]/2    # player center screen coordinate y
        for ray in range(rays):
            # Check horizonal lines
            dof = 0
            h_ray_dist = math.inf
            hx,hy = px,py
            aTan = -1/math.tan(ray_angle)
            if ray_angle>math.pi: # looking up
                ry = int(self.robot_pos[1]+0.5)*self.cell_height - 0.0001   # <- it took me 4 hours to figure out that subtracting 0.0001 was necessary to get this to work
                rx = (py-ry)*aTan + px
                yo = -self.cell_height
                xo = -yo*aTan
            elif ray_angle<math.pi: # looking down
                ry = int(self.robot_pos[1]+0.5+1)*self.cell_height
                rx = (py-ry)*aTan + px
                yo = self.cell_height
                xo = -yo*aTan
            else:   # looking straight left or right
                rx, ry = px, py
                dof = max(self.grid_rows, self.grid_cols)
            while dof < max(self.grid_rows, self.grid_cols):
                mx = int(rx // self.cell_width)
                my = int(ry // self.cell_height)
                mp = my*self.grid_cols+mx
                if(my>=self.grid_rows or my<0 or [mx,my] in self.wall_pos):
                    dof = max(self.grid_rows, self.grid_cols)
                    hx, hy = rx, ry
                    h_ray_dist = self.distance(px,py,hx,hy)
                else:
                    rx+=xo
                    ry+=yo
                    dof+=1
            # Check vertical lines
            dof = 0
            v_ray_dist = math.inf
            vx,vy = px,py
            nTan = -math.tan(ray_angle)
            if ray_angle>math.pi/2 and ray_angle<math.pi*3/2: # looking left
                rx = int(self.robot_pos[0]+0.5)*self.cell_width - 0.0001 
                ry = (px-rx)*nTan + py
                xo = -self.cell_width
                yo = -xo*nTan
            elif ray_angle>math.pi*3/2 or ray_angle<math.pi/2: # looking right
                rx = int(self.robot_pos[0]+0.5+1)*self.cell_width
                ry = (px-rx)*nTan + py
                xo = self.cell_width
                yo = -xo*nTan
            else:       # looking straight up or down
                rx, ry = px, py
                dof = max(self.grid_rows, self.grid_cols)
            while dof < max(self.grid_rows, self.grid_cols):
                mx = int(rx // self.cell_width)
                my = int(ry // self.cell_height)
                mp = my*self.grid_cols+mx
                if(mx>=self.grid_cols or mx<0 or [mx,my] in self.wall_pos):
                    dof = max(self.grid_rows, self.grid_cols)
                    vx, vy = rx, ry
                    v_ray_dist = self.distance(px,py,vx,vy)
                else:
                    rx+=xo
                    ry+=yo
                    dof+=1
            hit_pos = [hx,hy] if h_ray_dist<v_ray_dist else [vx,vy]
            pygame.draw.circle(self.window_surface,(0,0,255/rays*ray),hit_pos,5)
            pygame.draw.line(self.window_surface,(0,0,255/rays*ray),[px,py],hit_pos,width=2)
            ray_angle += (fov/(rays-1)) if rays>1 else 0
            
    def perform_action(self, robot_action:RobotAction) -> bool:
        self.last_action = robot_action

        # Rotate left
        if robot_action == RobotAction.LEFT:
            self.robot_facing_angle -= self.robot_turning_speed
            self.robot_facing_angle %= math.pi*2
            self.robot_delta_pos = [math.cos(self.robot_facing_angle)*self.robot_speed,math.sin(self.robot_facing_angle)*self.robot_speed]
        # Rotate right
        elif robot_action == RobotAction.RIGHT:
            self.robot_facing_angle += self.robot_turning_speed
            self.robot_facing_angle %= math.pi*2
            self.robot_delta_pos = [math.cos(self.robot_facing_angle)*self.robot_speed,math.sin(self.robot_facing_angle)*self.robot_speed]
        # Move forward
        elif robot_action == RobotAction.FORWARD:
            # self.robot_pos[0] += self.robot_delta_pos[0]
            # self.robot_pos[1] += self.robot_delta_pos[1]
            # desired_x = self.robot_pos[0] + math.cos(self.robot_facing_angle)*self.robot_speed
            # desired_y = self.robot_pos[1] + math.sin(self.robot_facing_angle)*self.robot_speed
            if self.is_valid_player_pos(self.robot_pos[0], self.robot_pos[1], self.robot_pos[0] + self.robot_delta_pos[0], self.robot_pos[1] + self.robot_delta_pos[1], self.grid_cols-1, self.grid_rows-1, self.wall_pos)[0]:
                self.robot_pos[0] += self.robot_delta_pos[0]
            if self.is_valid_player_pos(self.robot_pos[0], self.robot_pos[1], self.robot_pos[0] + self.robot_delta_pos[0], self.robot_pos[1] + self.robot_delta_pos[1], self.grid_cols-1, self.grid_rows-1, self.wall_pos)[1]:
                self.robot_pos[1] += self.robot_delta_pos[1]
        # Calculate which grid square the robot is in
        self.robot_grid_pos = [int(self.robot_pos[0]+0.5),int(self.robot_pos[1]+0.5)]
        # Pick up medicine if the robot moves over it
        if self.robot_grid_pos in self.medicine_pos:
            self.medicine_amt += 1
            self.medicine_pos.remove(self.robot_grid_pos)
        
        return (self.robot_grid_pos, self.target_pos, self.medicine_pos)

    def render(self):

        if self.window_surface is None:    
            pygame.init()
            if self.render_mode == 'human':
                pygame.display.init()
                self.window_surface = pygame.display.set_mode(self.window_size) 

        self._process_events()

        # Wipe the screen to get rid of any artifacts from last frame
        self.window_surface.fill((255,255,255))

        # Draw better background
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
        player_delta_center = ((self.robot_pos[0]+self.robot_delta_pos[0]/self.robot_speed)*self.cell_size[0]+self.cell_size[0]/2,(self.robot_pos[1]+self.robot_delta_pos[1]/self.robot_speed)*self.cell_size[1]+self.cell_size[1]/2)
        pygame.draw.circle(self.window_surface, "red", player_center, min(self.cell_size[0], self.cell_size[1])/2)
        pygame.draw.line(self.window_surface, "blue", player_center, player_delta_center)

        # Draw action display
        pygame.draw.rect(self.window_surface, pygame.Color(colors[len(colors)-1]), pygame.Rect(0,self.window_size[1]-self.action_info_height, self.window_size[0], self.action_info_height))
        text_img = self.action_font.render(f'Action: {self.last_action}', True, (0,0,0), pygame.Color(colors[len(colors)-1]))
        text_pos = (0, self.window_size[1] - self.action_info_height)
        self.window_surface.blit(text_img, text_pos) 

        self.raycast()

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
        if warehouseRobot.robot_grid_pos == warehouseRobot.target_pos:
            warehouseRobot.reset()
        
        if warehouseRobot.render_mode is not None:
            warehouseRobot.render()
