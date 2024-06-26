import pygame
import numpy as np
import gymnasium as gym

thief_img = ('thief.png')
police_img = ('policeman.png')
jail_open_door_img = ('jail_open_door.png')
wall_img = ('wall.png')
yellow_key_img = ('door_key.png')
thief_with_key_img = ('thief_with_key.png')
robber_in_jail_img = ('robber_in_jail.png')
two_robbers_img = ('two_robbers_released.png')
empty_jail_img = ('empty_jail.png')

wall_states = [(8,2),(4,6),(5,9),(2,5),(3,3),(0,7),(7,5)]
police_states = [(4,5),(1,7),(7,8)]
key_states = [(5,2)]
robber_in_jail = (5,0)
grid_size = 10

getting_key_reward = 100
releasing_robber_reward = 100
police_reward = -200
goal_reward = 500
else_reward = -1
class JailEnv(gym.Env):
    '''
    `JailEnv` is a custom environment where the agent (thief) has to escape the jail without getting caught by the police.
    The agent can move in four directions: up, down, right, and left.
    The agent can collect a key to open the cell of his friend (the second robber) to escape the jail.
    The agent can collect the key only once.
    The agent can get caught by the police.
    The agent can not pass through the walls.
    The agent can release the second robber from jail if the agent has the key.
    The agent wins the game by reaching the goal.
    The rewards are as follows:
    - The agent reaches the goal: +500
    - The agent gets caught by the police: -200
    - The agent collects the key of his friend cell: +100
    - The agent releases the second robber: +100
    - The agent moves: -1
    '''
    def __init__(self, grid_size=grid_size) -> None:
        super(JailEnv, self).__init__()
        self.grid_size = grid_size
        self.cell_size = 50

        self.thief_position = np.array([0, 0])
        self.reward = 0
        self.info = {}
        self.goal = np.array([grid_size-1,grid_size-1])
        self.done = False
        self.police_states = []
        self.wall_states = []
        self.key_states = []
        self.has_thief_collected_key = False
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32)
        self.robber_in_jail = robber_in_jail
        self.is_second_robber_released = False
        # Add police-states
        for i in range(len(police_states)):
            self.add_police_states(police_state_coordinates=(police_states[i][0],police_states[i][1]))

        # Add wall-states
        for i in range(len(wall_states)):
            self.add_wall_states(wall_state_coordinates=(wall_states[i][0],wall_states[i][1]))
        
        # Add keys-states
        for i in range(len(key_states)):
            self.add_key_states(key_state_coordinates=(key_states[i][0],key_states[i][1]))

        # Initialize the window:
        pygame.init()
        self.screen = pygame.display.set_mode((self.cell_size*self.grid_size, self.cell_size*self.grid_size))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("Jail Break")
        pass

    def calc_distance_to_goal(self):
        self.info["Distance to goal"] = np.sqrt(
            (self.thief_position[0]-self.goal[0])**2 +
            (self.thief_position[1]-self.goal[1])**2
            )
        return self.info["Distance to goal"]
    
    def reset(self):
        """ Method 1: .reset(): Reset the environment to the initial state
        Return the initial state of the environment and the info of the environment
        """
        self.thief_position = np.array([0, 0])
        self.done = False
        self.has_thief_collected_key = False
        self.is_second_robber_released = False
        self.reward = 0
        self.robber_in_jail = robber_in_jail
        for i in range(len(key_states)):
            self.add_key_states(key_state_coordinates=(key_states[i][0],key_states[i][1]))
           
        info = self.calc_distance_to_goal()

        return self.thief_position, info

    # Method 2: Add police states
    # ---------
    def add_police_states(self, police_state_coordinates):
        self.police_states.append(np.array(police_state_coordinates))

    def add_wall_states(self, wall_state_coordinates):
        self.wall_states.append(np.array(wall_state_coordinates))

    def add_key_states(self, key_state_coordinates):
        self.key_states.append(np.array(key_state_coordinates))

    def step(self, action):
        """ Method 2: .step(): Take a step in the environment based on the action taken by the agent
        Return the next state, reward, done, and info of the environment
        """
        ## Actions:
        ## --------
        initial_x=self.thief_position[0]
        initial_y=self.thief_position[1]
        up=0
        down=1
        right= 2
        left=3
        # Up:
        if action==up:
            can_go_up = self.thief_position[0] > 0
            if can_go_up:
                self.thief_position[0] -= 1

        # Down:
        elif action==down:
            can_go_down = self.thief_position[0] < self.grid_size-1
            if can_go_down:
                self.thief_position[0] += 1

        # Right:
        elif action==right:
            can_go_right = self.thief_position[1] < self.grid_size-1
            if can_go_right:
                self.thief_position[1] += 1

        # Left:
        elif action==left:
            can_go_left = self.thief_position[1] > 0
            if can_go_left:
                self.thief_position[1] -= 1  

        #If reached obsticle, undo last move
        if(True in [np.array_equal(self.thief_position, each_wall) for each_wall in self.wall_states]):
            self.thief_position[0] = initial_x
            self.thief_position[1] = initial_y
            # print('wall')

        ## Reward:
        ## -------
        if np.array_equal(self.thief_position, self.goal): # Check goal condition
            self.reward += goal_reward
            self.done = True
        # Check if the thief has run into the second robber and has key
        elif self.is_second_robber_released == False and np.array_equal(self.thief_position, self.robber_in_jail):
            self.reward += releasing_robber_reward
            self.is_second_robber_released = True
        #If reached key, remove key
        elif(not self.has_thief_collected_key and np.array_equal(self.thief_position, self.key_states[0])):
            self.key_states.pop(0)
            self.reward += getting_key_reward
            self.has_thief_collected_key = True
        elif True in [np.array_equal(self.thief_position, each_police) for each_police in self.police_states]: # Check police-states
            self.reward += police_reward
            self.done = True
            
         # Every other state    
        else:
            self.reward += else_reward
            self.done = False

        ## Info:
        ## -----
        info = self.has_thief_collected_key, self.is_second_robber_released
        return self.thief_position, self.reward, self.done, info

    def render(self):
        """ Method 3: .render(): Render the environment to visualize the current state of the environment"""
        # We make the background dark
        self.screen.fill((255,255,255))

        # Draw Grid lines:
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                grid = pygame.Rect(y*self.cell_size, x*self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (200,200,200), grid, 1)

        # Draw the Goal-state:
        goal = pygame.Rect(self.goal[1]*self.cell_size, self.goal[0]*self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (0,255,0), goal)
        open_door = pygame.image.load(jail_open_door_img)
        open_door = pygame.transform.scale(open_door, (self.cell_size, self.cell_size)) 
        self.screen.blit(open_door,(self.goal[1] *self.cell_size,self.goal[0] *self.cell_size))

        # Draw the police-states:
        for each_police in self.police_states:
            police = pygame.Rect(each_police[1]*self.cell_size, each_police[0]*self.cell_size, self.cell_size, self.cell_size)
            police = pygame.image.load(police_img)
            police = pygame.transform.scale(police, (self.cell_size, self.cell_size)) 
            self.screen.blit(police,(each_police[1] *self.cell_size,each_police[0] *self.cell_size))

        # Draw the wall-states:
        for each_wall in self.wall_states:
            wall = pygame.image.load(wall_img)
            wall = pygame.transform.scale(wall, (self.cell_size, self.cell_size)) 
            self.screen.blit(wall,(each_wall[1] *self.cell_size,each_wall[0] *self.cell_size)) 

        # Draw the robber-in-jail:
        if self.is_second_robber_released == False:
            robber_in_jail = pygame.image.load(robber_in_jail_img)
            robber_in_jail = pygame.transform.scale(robber_in_jail, (self.cell_size, self.cell_size))
            self.screen.blit(robber_in_jail,(self.robber_in_jail[1] *self.cell_size,self.robber_in_jail[0] *self.cell_size))    

        # Draw the key-states:
        if self.has_thief_collected_key == False:
            for each_key in self.key_states:
                key = pygame.image.load(yellow_key_img)
                key = pygame.transform.scale(key, (self.cell_size, self.cell_size)) 
                self.screen.blit(key,(each_key[1] *self.cell_size,each_key[0] *self.cell_size)) 


        # Draw the thief:
        if self.is_second_robber_released:
            thief = pygame.image.load(two_robbers_img)
        elif self.has_thief_collected_key:    
            thief = pygame.image.load(thief_with_key_img)
        else:    
            thief = pygame.image.load(thief_img)

        thief = pygame.transform.scale(thief, (self.cell_size, self.cell_size))
        self.screen.blit(thief,(self.thief_position[1] *self.cell_size,self.thief_position[0] *self.cell_size))
        # Update contents on the window:
        pygame.display.flip()
        pass

    def close(self):
        """ Method 4: .close(): Close the environment and free up the resources used by the environment"""
        pygame.quit()
        pass

        
    
# Function 1: Create an instance of the environment
# -----------
def create_env():
    # Create the environment:
    # -----------------------
    env = JailEnv()

    return env