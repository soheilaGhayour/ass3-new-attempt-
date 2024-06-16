# Imports:
# --------
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Function 1: Train Q-learning agent
# -----------
def train_q_learning(env,
                     no_episodes,
                     epsilon,
                     epsilon_min,
                     epsilon_decay,
                     alpha,
                     gamma,
                     q_table_save_path="q_table.npy"):

    # Initialize the Q-table:
    # -----------------------
    q_table = np.zeros((env.grid_size, env.grid_size, env.action_space.n))

    # Q-learning algorithm:
    # ---------------------
    #! Step 1: Run the algorithm for fixed number of episodes
    #! -------
    for episode in range(no_episodes):
        state, _ = env.reset()

        state = tuple(state)
        total_reward = 0

        #! Step 2: Take actions in the environment until "Done" flag is triggered
        #! -------
        while True:
            #! Step 3: Define your Exploration vs. Exploitation
            #! -------
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit

            next_state, reward, done, _ = env.step(action)
            # env.render()

            next_state = tuple(next_state)
            total_reward += reward

            #! Step 4: Update the Q-values using the Q-value update rule
            #! -------
            q_table[state][action] = q_table[state][action] + alpha * \
                (reward + gamma *
                 np.max(q_table[next_state]) - q_table[state][action])

            state = next_state

            #! Step 5: Stop the episode if the agent reaches Goal or Hell-states
            #! -------
            if done:
                break

        #! Step 6: Perform epsilon decay
        #! -------
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    #! Step 7: Close the environment window
    #! -------
    env.close()
    print("Training finished.\n")

    #! Step 8: Save the trained Q-table
    #! -------
    np.save(q_table_save_path, q_table)
    print("Saved the Q-table.")


# Function 2: Visualize the Q-table
# -----------
def visualize_q_table(police_states=[(2, 1), (0, 4)],
                      goal_coordinates=(9, 9),
                      wall_states=[(8,2),(4,6),(5,9),(2,5),(3,3),(0,7),(7,5)],
                      key_states=[(2,2)],
                      robber_in_jail=(5,0),
                      actions=["Up", "Down", "Right", "Left"],
                      q_values_path="q_table.npy"):

    # Load the Q-table:
    # -----------------
    try:
        q_table = np.load(q_values_path)

        # Create subplots for each action:
        # --------------------------------
        _, axes = plt.subplots(1, 4, figsize=(20, 5))

        for i, action in enumerate(actions):
            ax = axes[i]
            heatmap_data = q_table[:, :, i].copy()

            # Mask the goal state's Q-value for visualization:
            # ------------------------------------------------
            mask = np.zeros_like(heatmap_data, dtype=bool)
            # mask[goal_coordinates] = True
            
            # for wall in wall_states:
            #     mask[wall] = True
            # for police in police_states:
            #     mask[police] = True
            # for key in key_states:
            #     mask[key] = True    

            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis",
                        ax=ax, cbar=False, mask=mask, annot_kws={"size": 9})

            # Denote Goal and Hell states:
            # ----------------------------
            ax.text(goal_coordinates[1] + 0.5, goal_coordinates[0] + 0.5, 'G', color='green',
                    ha='center', va='center', weight='bold', fontsize=14)
            for key in key_states:
                ax.text(key[1] + 0.5, key[0] + 0.5, 'K', color='yellow',
                        ha='center', va='center', weight='bold', fontsize=14)
            for police in police_states:
                ax.text(police[1] + 0.5, police[0] + 0.5, 'P', color='red',
                        ha='center', va='center', weight='bold', fontsize=14)    
            for wall in wall_states:
                ax.text(wall[1] + 0.5, wall[0] + 0.5, 'W', color='black',
                        ha='center', va='center', weight='bold', fontsize=14)
            ax.text(robber_in_jail[1] + 0.5, robber_in_jail[0] + 0.5, 'J', color='blue',
                    ha='center', va='center', weight='bold', fontsize=14)    

            ax.set_title(f'Action: {action}')

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("No saved Q-table was found. Please train the Q-learning agent first or check your path.")
