# Imports:
# --------
import gymnasium as gym
import numpy as np
import pygame
import sys


# Class 1: Define a custom environment
# --------
class PadmEnv(gym.Env):
    def __init__(self, grid_size=5, goal_coordinates=(4, 4)) -> None:
        super(PadmEnv, self).__init__()
        self.grid_size = grid_size
        self.cell_size = 100
        self.state = None
        self.reward = 0
        self.info = {}
        self.goal = np.array(goal_coordinates)
        self.done = False
        self.hell_states = []

        # Action-space:
        self.action_space = gym.spaces.Discrete(4)

        # Observation space:
        self.observation_space = gym.spaces.Box(
            low=0, high=grid_size-1, shape=(2,), dtype=np.int32)

        # Initialize the window:
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.cell_size*self.grid_size, self.cell_size*self.grid_size))

    # Method 1: .reset()
    # ---------

    def reset(self):
        """
        Everything must be reset
        """
        self.state = np.array([0, 0])
        self.done = False
        self.reward = 0

        self.info["Distance to goal"] = np.sqrt(
            (self.state[0]-self.goal[0])**2 +
            (self.state[1]-self.goal[1])**2
        )

        return self.state, self.info

    # Method 2: Add hell states
    # ---------

    def add_hell_states(self, hell_state_coordinates):
        self.hell_states.append(np.array(hell_state_coordinates))

    # Method 3: .step()
    # ---------

    def step(self, action):
        # Actions:
        # --------
        # Up:
        if action == 0 and self.state[0] > 0:
            self.state[0] -= 1

        # Down:
        if action == 1 and self.state[0] < self.grid_size-1:
            self.state[0] += 1

        # Right:
        if action == 2 and self.state[1] < self.grid_size-1:
            self.state[1] += 1

        # Left:
        if action == 3 and self.state[1] > 0:
            self.state[1] -= 1

        # Reward:
        # -------
        if np.array_equal(self.state, self.goal):  # Check goal condition
            self.reward += 10
            self.done = True
        # Check hell-states
        elif True in [np.array_equal(self.state, each_hell) for each_hell in self.hell_states]:
            self.reward += -1
            self.done = True
        else:  # Every other state
            self.reward += 0
            self.done = False

        # Info:
        # -----
        self.info["Distance to goal"] = np.sqrt(
            (self.state[0]-self.goal[0])**2 +
            (self.state[1]-self.goal[1])**2
        )

        return self.state, self.reward, self.done, self.info

    # Method 3: .render()
    # ---------

    def render(self):
        # Code for closing the window:
        for event in pygame.event.get():
            if event == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # We make the background White
        self.screen.fill((255, 255, 255))

        # Draw Grid lines:
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                grid = pygame.Rect(
                    y*self.cell_size, x*self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (200, 200, 200), grid, 1)

        # Draw the Goal-state:
        goal = pygame.Rect(self.goal[1]*self.cell_size, self.goal[0]
                           * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (0, 255, 0), goal)

        # Draw the hell-states:
        for each_hell in self.hell_states:
            hell = pygame.Rect(
                each_hell[1]*self.cell_size, each_hell[0]*self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (255, 0, 0), hell)

        # Draw the agent:
        agent = pygame.Rect(self.state[1]*self.cell_size, self.state[0]
                            * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (0, 0, 0), agent)

        # Update contents on the window:
        pygame.display.flip()

    # Method 4: .close()
    # ---------

    def close(self):
        pygame.quit()


# Function 1: Create an instance of the environment
# -----------
def create_env(goal_coordinates,
               hell_state_coordinates):
    # Create the environment:
    # -----------------------
    env = PadmEnv(goal_coordinates=goal_coordinates)

    for i in range(len(hell_state_coordinates)):
        env.add_hell_states(hell_state_coordinates=hell_state_coordinates[i])

    return env
