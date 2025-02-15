'''
Custom Gym environment
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
'''
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import v0_warehouse_robot as wr
import numpy as np
import math
import pygame
import constants as const

# Define the environment ID
env_id = 'warehouse-robot-v0'   # call it whatever you want

# Check if the environment is already registered
if env_id not in gym.envs.registry:
    register(
        id=env_id,                                              
        entry_point='v0_warehouse_robot_env:WarehouseRobotEnv', # module_name:class_name
    )
# Implement our own gym env, must inherit from gym.Env
# https://gymnasium.farama.org/api/env/
class WarehouseRobotEnv(gym.Env):
    # metadata is a required attribute
    # render_modes in our environment is either None or 'human'.
    # render_fps is not used in our env, but we are require to declare a non-zero value.
    metadata = {"render_modes": ["human"], 'render_fps': const.FPS}

    def __init__(self, grid_rows=const.GRID_ROWS if not const.MAP else len(const.MAP), grid_cols=const.GRID_COLS if not const.MAP else len(const.MAP[0]), render_mode=None):

        self.grid_rows=grid_rows
        self.grid_cols=grid_cols
        self.render_mode = render_mode

        # Initialize the WarehouseRobot problem
        self.warehouse_robot = wr.WarehouseRobot(self.grid_rows, self.grid_cols, self.metadata['render_fps'], self.render_mode)

        # Gym requires defining the action space. The action space is robot's set of possible actions.
        # Training code can call action_space.sample() to randomly select an action. 
        self.action_space = spaces.Discrete(len(wr.RobotAction))

        # Gym requires defining the observation space. The observation space consists of the robot's and target's set of possible positions.
        # The observation space is used to validate the observation returned by reset() and step().
        # The current approach is to build a 1-dimensional array containing all relevant data like player and wall positions
        low = np.array([])
        low = np.append(low, 0.0)    # min robot facing angle
        low = np.append(low, 0.0)    # min robot row position
        low = np.append(low, 0.0)    # min robot col position
        low = np.append(low, 0.0)    # min target row position
        low = np.append(low, 0.0)    # min target col position
        for i in range(const.RAYS):
            low = np.append(low, 0.0)    # min value of tile in sight
        for i in range(const.RAYS):
            low = np.append(low, 0.0)    # min length of ray
        
        high = np.array([])
        high = np.append(high, math.pi * 2)    # max robot facing angle
        high = np.append(high, 1.0)    # max robot row position
        high = np.append(high, 1.0)    # max robot col position
        high = np.append(high, 1.0)    # max target row position
        high = np.append(high, 1.0)    # max target col position
        for i in range(const.RAYS):
            high = np.append(high, 1.0)    # max value of tile in sight
        for i in range(const.RAYS):
            high = np.append(high, math.sqrt(self.grid_rows*self.grid_rows+self.grid_cols*self.grid_cols)*const.CELL_WIDTH)    # max length of ray

        # Use a 1D vector: [robot_row_pos, robot_col_pos, robot_facing_angle, target_row_pos, target_col_pos] 
        # self.observation_space = gym.spaces.Box(low=low, high=high, shape =(3+(2*const.RAYS),), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, shape =(5+(2*const.RAYS),), dtype=np.float32)

        self.max_ray_dist = math.sqrt((self.grid_rows-0.5)*(self.grid_rows-0.5)+(self.grid_cols-0.5)*(self.grid_cols-0.5))*const.CELL_WIDTH

    def get_obs(self):
        obs = np.concatenate((
            np.array([self.warehouse_robot.robot_facing_angle]), # float from 0-6.28
            np.array([self.warehouse_robot.robot_pos[0]/(self.grid_cols-1)]),    # normalized x position of robot
            np.array([self.warehouse_robot.robot_pos[1]/(self.grid_rows-1)]),    # normalized y position of robot
            np.array([self.warehouse_robot.target_pos[0]/(self.grid_cols-1)]),    # normalized x position of target
            np.array([self.warehouse_robot.target_pos[1]/(self.grid_rows-1)]),    # normalized y position of target
            np.array((self.warehouse_robot.raycast(rays=const.RAYS,fov=const.FOV))[0]), # list of raycast values
            np.array(([dist/self.max_ray_dist for dist in self.warehouse_robot.raycast(rays=const.RAYS,fov=const.FOV)[1]])) # list of normalized distance values
            ))
        return obs 
        
    # Gym required function (and parameters) to reset the environment
    def reset(self, seed=None, options=None):
        super().reset() # gym requires this call to control randomness and reproduce scenarios.

        self.num_steps = 0
        self.reward = 0

        # Reset the WarehouseRobot. Optionally, pass in seed control randomness and reproduce scenarios.
        self.warehouse_robot.reset()

        # Construct the observation state:
        obs = self.get_obs()

        # Additional info to return. For debugging or whatever.
        info = {}

        # Render environment
        if(self.render_mode=='human'):
            self.render()

        # Return observation and info
        obs = np.array(obs, dtype=np.float32)   # Hack to make unexpected type error to go away
        return obs, info

    # Gym required function (and parameters) to perform an action
    def step(self, action):

        # Perform action
        robot_grid_pos, target_pos, tiles_hit = self.warehouse_robot.perform_action(wr.RobotAction(action))

        # Determine reward and termination
        terminated = False
        truncated = False
        if robot_grid_pos == target_pos:
            terminated = True
            self.reward += 10 #10 og
        elif self.num_steps > const.MAX_STEPS:
            truncated = True
            self.reward += sum(tiles_hit)/const.RAYS * 10
        else:
            self.reward += -10/const.MAX_STEPS 
        
        reward = self.reward

        # Construct the observation state: 
        obs = self.get_obs()
        
        # Additional info to return. For debugging or whatever.
        info = {}

        # Render environment
        if(self.render_mode=='human'):
            self.render()

        # Increment number of steps
        self.num_steps += 1
        
        # Return observation, reward, terminated, truncated, info
        obs = np.array(obs, dtype=np.float32)       # Hack to make unexpected type error to go away
        return obs, reward, terminated, truncated, info

    # Gym required function to render environment
    def render(self):
        self.warehouse_robot.render()

# For unit testing
if __name__=="__main__":
    env = gym.make('warehouse-robot-v0', render_mode='human')

    # Reset environment
    obs = env.reset()[0]

    running=True
    while(running):
        # Manually run using keyboard
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        terminated = False
        truncated = False
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]:
            man_action = 0
            obs, reward, terminated, truncated, _ = env.step(man_action)
        if keys[pygame.K_w]:
            man_action = 1
            obs, reward, terminated, truncated, _ = env.step(man_action)
        if keys[pygame.K_d]:
            man_action = 2
            obs, reward, terminated, truncated, _ = env.step(man_action)

        if(terminated or truncated):
            obs = env.reset()[0]
