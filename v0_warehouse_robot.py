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

    def generate_map(self, map=const.MAP):
        self.wall_pos=[]
        self.medicine_pos=[]
        self.target_pos=[]
        for y in range(len(const.MAP)):
            for x in range(len(const.MAP[y])):
                if const.MAP[y][x] == 1:
                    self.wall_pos.append([x,y])
                elif const.MAP[y][x] == 2:
                    self.medicine_pos.append([x,y])
                elif const.MAP[y][x] == 3:
                    self.target_pos=[x,y]


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
        self.robot_delta_pos = [math.cos(self.robot_facing_angle)*self.robot_speed,math.sin(self.robot_facing_angle)*self.robot_speed]

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
        self.medicine_pos.append(self.target_pos)

        # Store values for reset()
        self.reset_robot_pos = self.robot_pos.copy()
        self.reset_robot_facing_angle = self.robot_facing_angle
        self.reset_medicine_pos = self.medicine_pos.copy()

        # Ensure robot never faces target (so that it doesn't learn to hold forward and get lucky)
        angle_to_target = math.atan2(self.robot_pos[1]-self.medicine_pos[0][1],self.robot_pos[0]-self.medicine_pos[0][0])
        c = angle_to_target if angle_to_target > 0 else math.pi*2 + angle_to_target
        # Calculate values for observation space
        self.alignment = (math.pi - abs(abs(self.robot_facing_angle-c)-math.pi))/math.pi    # How close the agent is to facing the target
        if self.alignment > 0.5:
            self.robot_facing_angle -= 3.1415926535
            self.robot_facing_angle %= 2*math.pi
            angle_to_target = math.atan2(self.robot_pos[1]-self.medicine_pos[0][1],self.robot_pos[0]-self.medicine_pos[0][0])
            c = angle_to_target if angle_to_target > 0 else math.pi*2 + angle_to_target
            self.alignment = (math.pi - abs(abs(self.robot_facing_angle-c)-math.pi))/math.pi    # How close the agent is to facing the target
        self.inv_dist = 1-self.distance(self.robot_pos[0],self.robot_pos[1],self.medicine_pos[0][0],self.medicine_pos[0][1])/self.distance(self.reset_robot_pos[0],self.reset_robot_pos[1],self.medicine_pos[0][0],self.medicine_pos[0][1])  # How close the agent is to the target
        self.robot_delta_pos = [math.cos(self.robot_facing_angle)*self.robot_speed,math.sin(self.robot_facing_angle)*self.robot_speed]

    def reset(self):
        if const.IS_RANDOM:
            self.generate_hospital(seed=None)
        if len(self.medicine_pos) == 1:   # Do not reset the robot's position if there is still medicine
            self.robot_pos = self.reset_robot_pos.copy()
            self.robot_facing_angle = self.reset_robot_facing_angle
        self.robot_delta_pos = [math.cos(self.robot_facing_angle)*self.robot_speed,math.sin(self.robot_facing_angle)*self.robot_speed]
        self.medicine_pos = self.reset_medicine_pos.copy()

        # Ensure robot never faces target (so that it doesn't learn to hold forward and get lucky)
        angle_to_target = math.atan2(self.robot_pos[1]-self.medicine_pos[0][1],self.robot_pos[0]-self.medicine_pos[0][0])
        c = angle_to_target if angle_to_target > 0 else math.pi*2 + angle_to_target
        # Calculate values for observation space
        self.alignment = (math.pi - abs(abs(self.robot_facing_angle-c)-math.pi))/math.pi    # How close the agent is to facing the target
        if self.alignment > 0.5 and not self.medicine_pos:
            self.robot_facing_angle -= 3.1415926535
            self.robot_facing_angle %= 2*math.pi
            angle_to_target = math.atan2(self.robot_pos[1]-self.medicine_pos[0][1],self.robot_pos[0]-self.medicine_pos[0][0])
            c = angle_to_target if angle_to_target > 0 else math.pi*2 + angle_to_target
            self.alignment = (math.pi - abs(abs(self.robot_facing_angle-c)-math.pi))/math.pi    # How close the agent is to facing the target
            self.robot_delta_pos = [math.cos(self.robot_facing_angle)*self.robot_speed,math.sin(self.robot_facing_angle)*self.robot_speed]
        self.inv_dist = 1-self.distance(self.robot_pos[0],self.robot_pos[1],self.medicine_pos[0][0],self.medicine_pos[0][1])/self.distance(self.reset_robot_pos[0],self.reset_robot_pos[1],self.medicine_pos[0][0],self.medicine_pos[0][1])  # How close the agent is to the target

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
    
    def distance(self, ax, ay, bx, by):
        return math.sqrt((bx-ax)*(bx-ax)+(by-ay)*(by-ay))
    
    def raycast(self, rays=1, fov=180, draw_rays=False):
        distances = []
        self.fov = fov * 0.0174533 
        self.rays = rays
        self.draw_rays = draw_rays
        ray_angle = self.robot_facing_angle - self.fov/2 if  self.rays > 1 else self.robot_facing_angle
        ray_angle %= math.pi*2 
        px = self.robot_pos[0]*self.cell_size[0]+self.cell_size[0]/2    # player center screen coordinate x
        py = self.robot_pos[1]*self.cell_size[1]+self.cell_size[1]/2    # player center screen coordinate y
        for ray in range(self.rays):
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
                if(mx>=self.grid_cols or mx<0 or [mx,my] in self.wall_pos):
                    dof = max(self.grid_rows, self.grid_cols)
                    vx, vy = rx, ry
                    v_ray_dist = self.distance(px,py,vx,vy)
                else:
                    rx+=xo
                    ry+=yo
                    dof+=1
            hit_pos = [hx,hy] if h_ray_dist<v_ray_dist else [vx,vy]
            distances.append(min(h_ray_dist,v_ray_dist))
            # Visualise raycast
            if self.draw_rays:
                pygame.draw.circle(self.window_surface,(255-(255/ self.rays*ray),0,0),hit_pos,5)
                pygame.draw.line(self.window_surface,(255-(255/ self.rays*ray),0,0),[px,py],hit_pos,width=2)
            ray_angle += (self.fov/( self.rays-1)) if  self.rays>1 else 0
            ray_angle %= math.pi*2
        return(distances)
            
    def perform_action(self, robot_action:RobotAction):
        self.last_action = robot_action
        moved=False

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
            self.robot_facing_angle += 0
            self.robot_facing_angle %= math.pi*2
            self.robot_delta_pos = [math.cos(self.robot_facing_angle)*self.robot_speed,math.sin(self.robot_facing_angle)*self.robot_speed]
            if self.is_valid_player_pos(self.robot_pos[0], self.robot_pos[1], self.robot_pos[0] + self.robot_delta_pos[0], self.robot_pos[1] + self.robot_delta_pos[1], self.grid_cols-1, self.grid_rows-1, self.wall_pos)[0]:
                self.robot_pos[0] += self.robot_delta_pos[0]
            if self.is_valid_player_pos(self.robot_pos[0], self.robot_pos[1], self.robot_pos[0] + self.robot_delta_pos[0], self.robot_pos[1] + self.robot_delta_pos[1], self.grid_cols-1, self.grid_rows-1, self.wall_pos)[1]:
                self.robot_pos[1] += self.robot_delta_pos[1]
            moved=True
        # Calculate which grid square the robot is in
        self.robot_grid_pos = [int(self.robot_pos[0]+0.5),int(self.robot_pos[1]+0.5)]

        # Pick up medicine if the robot moves over it
        if self.robot_grid_pos == self.medicine_pos[0] and len(self.medicine_pos) > 1: 
            self.medicine_pos.remove(self.robot_grid_pos)

        angle_to_target = math.atan2(self.robot_pos[1]-self.medicine_pos[0][1],self.robot_pos[0]-self.medicine_pos[0][0])
        c = angle_to_target if angle_to_target > 0 else math.pi*2 + angle_to_target
        self.alignment = (math.pi - abs(abs(self.robot_facing_angle-c)-math.pi))/math.pi    # How close the agent is to facing the target
        self.inv_dist = 1-self.distance(self.robot_pos[0],self.robot_pos[1],self.medicine_pos[0][0],self.medicine_pos[0][1])/self.distance(self.reset_robot_pos[0],self.reset_robot_pos[1],self.medicine_pos[0][0],self.medicine_pos[0][1])  # How close the agent is to the target

        return (self.robot_grid_pos, self.medicine_pos[0], self.alignment, self.inv_dist, moved and self.alignment>0.5) 

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

        # Draw medicine
        num_med = len(self.medicine_pos)
        for medicine in (self.medicine_pos):
            pygame.draw.rect(self.window_surface, (0,150,0), pygame.Rect(medicine[0]*self.cell_size[0], medicine[1]*self.cell_size[1], self.cell_size[0], self.cell_size[1]))
            pygame.draw.circle(self.window_surface, (150,255,150), (medicine[0]*self.cell_size[0]+self.cell_size[0]/2,medicine[1]*self.cell_size[1]+self.cell_size[1]/2), (min(self.cell_size[0], self.cell_size[1]))/4)
            if num_med == len(self.medicine_pos):
                pygame.draw.rect(self.window_surface, (255,150,0), pygame.Rect(medicine[0]*self.cell_size[0], medicine[1]*self.cell_size[1], self.cell_size[0], self.cell_size[1]))
                pygame.draw.circle(self.window_surface, (255,255,150), (medicine[0]*self.cell_size[0]+self.cell_size[0]/2,medicine[1]*self.cell_size[1]+self.cell_size[1]/2), (min(self.cell_size[0], self.cell_size[1]))/4)
            num_med -= 1

        # Draw target
        # pygame.draw.rect(self.window_surface, "blue", pygame.Rect(self.target_pos[0]*self.cell_size[0], self.target_pos[1]*self.cell_size[1], self.cell_size[0], self.cell_size[1]))
        # pygame.draw.rect(self.window_surface, (150,150,255), pygame.Rect(self.target_pos[0]*self.cell_size[0]+self.grid_line_width, self.target_pos[1]*self.cell_size[1]+self.grid_line_width, self.cell_size[0]-self.grid_line_width*2, self.cell_size[1]-self.grid_line_width*2))

        # Draw walls
        for wall in (self.wall_pos):
            pygame.draw.rect(self.window_surface, "black", pygame.Rect(wall[0]*self.cell_size[0], wall[1]*self.cell_size[1], self.cell_size[0], self.cell_size[1]))

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

        # Visualize raycast
        # if const.RENDER_DURING_TRAINING:
        #     self.raycast(rays=5,draw_rays=True)

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

    running=True
    while(running):
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
        if keys[pygame.K_r]:
            warehouseRobot.reset()
        if warehouseRobot.robot_grid_pos == warehouseRobot.medicine_pos[0]:
            warehouseRobot.reset()
        
        if warehouseRobot.render_mode is not None:
            warehouseRobot.render()
