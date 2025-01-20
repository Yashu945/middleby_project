import numpy as np

# Define the maze as a numpy array
maze = np.array([
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
])

# Start and goal positions
start_position = (0, 0)
goal_position = (4, 4)

def initialize_q_table(size, actions):
    return np.zeros((size, size, len(actions)))

actions = ['up', 'down', 'left', 'right']  # Possible actions
q_table = initialize_q_table(maze.shape[0], actions)

# Hyperparameters
learning_rate = 0.1
discount_factor = 0.99
exploration_rate = 1.0
max_exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 0.01

# Q-learning update rule
def update_q_table(old_state, action, reward, new_state, done):
    old_q_value = q_table[old_state][action]
    future_optimal_value = np.max(q_table[new_state]) if not done else 0
    new_q_value = (1 - learning_rate) * old_q_value + learning_rate * (reward + discount_factor * future_optimal_value)
    q_table[old_state][action] = new_q_value

import random

def choose_action(state, exploration_rate):
    if random.uniform(0, 1) < exploration_rate:
        # Explore: Choose an action index randomly instead of the action itself
        return random.choice(range(len(actions)))  
    else:
        return np.argmax(q_table[state])  # Exploit best known action

def get_reward(new_state):
    if new_state == goal_position:
        return 100  # Reward for reaching the goal
    elif maze[new_state] == 1:
        return -100  # Penalty for hitting an obstacle
    else:
        return -1  # Small penalty for each move

def move_agent(state, action):
    """Moves the agent in the maze based on the chosen action.

    Args:
        state: The current state (position) of the agent in the maze (tuple).
        action: The action chosen by the agent (string).

    Returns:
        The new state (position) of the agent after performing the action (tuple).
    """
    row, col = state
    if action == 'up':
        new_row, new_col = row - 1, col
    elif action == 'down':
        new_row, new_col = row + 1, col
    elif action == 'left':
        new_row, new_col = row, col - 1
    elif action == 'right':
        new_row, new_col = row, col + 1
    else:
        raise ValueError(f"Invalid action: {action}")

    # Check if the new state is within the maze boundaries
    if 0 <= new_row < maze.shape[0] and 0 <= new_col < maze.shape[1]:
        return new_row, new_col
    else:
        return state  # Stay in the current state if the move is invalid

num_episodes = 1000
for episode in range(num_episodes):
    state = start_position
    done = False
    while not done:
        action_index = choose_action(state, exploration_rate)
        new_state = move_agent(state, actions[action_index])  
        reward = get_reward(new_state)
        done = new_state == goal_position or maze[new_state] == 1
        update_q_table(state, action_index, reward, new_state, done)
        state = new_state

    # Decay exploration rate
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
    
# Print the Q-table
print("Q-table:")
print(q_table)

def visualize_path(start, goal, q_table):
    path = []
    state = start
    while state != goal:
        action_index = np.argmax(q_table[state])
        state = move_agent(state, actions[action_index])
        path.append(state)
        if state == goal:
            print("Goal reached!")
            break
    return path

path = visualize_path(start_position, goal_position, q_table)
print("Path from start to goal:", path)
