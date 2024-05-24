# reinforcement-learning-simulation
3 agents in a 2d world

# RL Path Finding with 3 Agents in a 2D world - Team URSA

The program runs 4 different experiments of a multi-agent pathfinding algorithm using reinforcement learning with Q-learning and SARSA implementations.

## Installation

Open the program in IDE
Install dependencies: ‘pip install numpy’

## Usage

In terminal make sure you are in the correct directory and type : ‘py {program_name}.py’ (please note that ‘py’ could be different depending on OS or IDE, also if program gets renamed make sure you are using the correct name) 
Users will be prompted to input an experiment number based on a menu. For a certain experiment, type the menu item EXACTLY like how it is listed in the menu.
The experiment will then be printed in a newly created output file in the same directory as the program (it is suggested to run the program in its own separate folder if you plan on keeping the experiment files and it is important to save or rename each run of an experiment if you do not want the file to be overwritten by the next run).

## Features

Multi-agent Pathfinding: Through the Agent class, we encapsulated individual agent functionality with a grid-based environment. Given dynamic movement within the grid while avoiding occupied spaces by other agents. Real-time scoring mechanism is reflecting rewards and penalties incurred by agent actions. Ability to pick up and drop off blocks within the environment.
Q-learning and SARSA: Implementation of Q-learning algorithm for optimal action selection. Implementation of SARSA learning algorithm for policy iteration. Q-table storage for efficient state-action value updating. Support of random, greedy, and exploit action selection strategies.
Customizable Parameters: ‘alpha’, ‘gamma’, and ‘epsilon’ for Q-Learning and SARSA algorithms
