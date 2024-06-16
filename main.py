# Imports:
# --------
from assignment1 import create_env
from Q_learning import train_q_learning, visualize_q_table

# User definitions:
# -----------------
train = True
visualize_results = True

learning_rate = 0.01  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.1  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate for exploration
no_episodes = 1_000  # Number of episodes

goal_coordinates = (9, 9)
# Define all hell state coordinates as a tuple within a list

wall_states = [(8,2),(4,6),(5,9),(2,5),(3,3),(0,7),(7,5)]
police_states = [(4,5),(1,7),(7,8)]
key_states = [(2,2)]
robber_in_jail = (5,0)


# Execute:
# --------
if train:
    # Create an instance of the environment:
    # --------------------------------------
    env = create_env()

    # Train a Q-learning agent:
    # -------------------------
    train_q_learning(env=env,
                     no_episodes=no_episodes,
                     epsilon=epsilon,
                     epsilon_min=epsilon_min,
                     epsilon_decay=epsilon_decay,
                     alpha=learning_rate,
                     gamma=gamma)

if visualize_results:
    # Visualize the Q-table:
    # ----------------------
    visualize_q_table(police_states=police_states,
                      goal_coordinates=goal_coordinates,
                      wall_states=wall_states,
                      key_states=key_states,
                      robber_in_jail=robber_in_jail,
                      q_values_path="q_table.npy")
