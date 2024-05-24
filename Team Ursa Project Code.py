import numpy as np
import random

pickupLocations = {(0, 4): 5, (1, 3): 5, (4, 1): 5} #set initial pickup Locations
dropoffLocations = {(0, 0): 0, (2, 0): 0, (3, 4): 0} #set up inital dropoff Locations
world_size = 5 #set grid size 5x5

class Agent:
    #This module represents an individual agent in the environment
    def __init__(self, name, position): # initializes the agent with a name, position, a flag indicating if it has a block, an initial score, and reward for each state
        self.name = name
        self.position = position
        self.have_block = False
        self.score = 0
        self.reward = 0

    def move(self, action, agents): # responsible for moving the agent within the environment based on specified action
        new_x, new_y = self.position[0] + action[0], self.position[1] + action[1]
        if 0 <= new_x < world_size and 0 <= new_y < world_size:
            # check if another agent occupies the target position
            for agent in agents:
                if agent.position == (new_x, new_y):
                    # another agent occupies the target position, try a different direction
                    return False
            
            self.position = (new_x, new_y)
            self.score -= 1
            self.reward = -1

            # perform pickup and dropoff, if applicable
            self.pickup()
            self.dropoff()

            return True

        else:
            # agent cannot move outside the bounds of the world, try a different direction
            return False
    
    def can_pickup(self, state): # checks if agent is at a pickup location, if the location has blocks left, and if the agent is holding a block
        return state in pickupLocations and pickupLocations[state] > 0 and not self.have_block
    
    def can_dropoff(self, state): # checks if agent is at a dropoff location, if the location has blocks left, and if the agent is holding a block
        return state in dropoffLocations and dropoffLocations[state] < 5 and self.have_block
    
    def pickup(self): #executes pickup action if applicable, giving the agent a reward
        if self.can_pickup(self.position):
            self.have_block = True
            self.score += 14
            self.reward = 13
            pickupLocations[self.position] -= 1  # Remove one package from the pickup location

    def dropoff(self): #executes dropoff action if applicable, giving the agent a reward
        if self.can_dropoff(self.position):
            self.have_block = False
            self.score += 14
            self.reward = 13
            dropoffLocations[self.position] += 1  # Remove one package from the dropoff location


class QLearningAgent:
    #This module defines a Q-learning agent that learns to act optimally by estimating the quality of taking actions in different states
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1): #initializes the Q-learning agent with parameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def get_q_value(self, state, action): #returns Q-value for a given state-action pair. If not present, initializes with value of 0.
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0
        return self.q_table[state][action]
    
    def get_action(self, state, actions): #selects an action based on an epsilon-greedy strategy
        if random.uniform(0, 1) < self.epsilon: 
            return random.choice(actions)  # random action
        else:
            if state not in self.q_table:
                self.q_table[state] = {action: 0 for action in actions}
            return max(self.q_table[state], key=self.q_table[state].get)  # greedy action
        
    def PRandom(self, state, actions): #chooses an action at random
        return random.choice(actions)
    
    def PGreedy(self, state, actions):
        if random.uniform(0, 1) < self.epsilon: #explores with a probability, otherwise exploits by choosing the action with the highest Q-value for the state
            return random.choice(actions)  # random action exploration
        else:
            if state not in self.q_table:
                self.q_table[state] = {action: 0 for action in actions}

            max_q_value = max(self.q_table[state].values())
            best_actions = [action for action, q_value in self.q_table[state].items() if q_value == max_q_value]

            if len(best_actions) > 1:  # if there are ties
                # break ties by rolling a dice
                action = random.choice(best_actions)
            else:
                action = best_actions[0]  # choose the only action if there are no ties

            return action    
        
         
    def PExploit(self, state, actions): #exploits with a probability of 80% to choose from the action with the highest Q-value, otherwise choose an action at random

        if random.uniform(0, 1) < 0.80:
            max_q_value = max(self.q_table[state].values())
            best_actions = [action for action, q_value in self.q_table[state].items() if q_value == max_q_value]
            
            return random.choice(best_actions)
        else:
            return random.choice(actions)

    def update_q_value(self, state, action, reward, next_state, next_action): #updates Q-value for a state-action pair based on observed reward and estimated future rewards
        if state not in self.q_table:
            self.q_table[state] = {}
        if next_state not in self.q_table:
            self.q_table[next_state] = {}

        if action not in self.q_table[state]:
            self.q_table[state][action] = 0
        if next_action not in self.q_table[next_state]:
            self.q_table[next_state][next_action] = 0

        self.q_table[state][action] += self.alpha * (
                    reward + self.gamma * self.q_table[next_state][next_action] - self.q_table[state].get(action, 0))
        
###########################################

class SARSA: #defines SARSA agent, learns Q-values for state-action pairs while following policy derived from current Q-values
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1): #initializes the SARSA agent with parameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def get_q_value(self, state, action): #returns Q-value for given state-action pair, if pair is not present, initializes with a value of 0
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0
        return self.q_table[state][action]

    def get_action(self, state, actions): #selects an action based on epsislon-greedy strategy
        if random.uniform(0, 1) < self.epsilon: 
            return random.choice(actions)  # random action
        else:
            if state not in self.q_table:
                self.q_table[state] = {action: 0 for action in actions}
            return max(self.q_table[state], key=self.q_table[state].get)  # greedy action
        
    def PRandom(self, state, actions): #chooses action at random
        return random.choice(actions)
    
    def PGreedy(self, state, actions): #explores probability of epsilon, otherwise exploits by choosing the action with the highest Q-value for the state
        if random.uniform(0, 1) < self.epsilon: 
            return random.choice(actions)  # random action exploration
        else:
            if state not in self.q_table:
                self.q_table[state] = {action: 0 for action in actions}

            max_q_value = max(self.q_table[state].values())
            best_actions = [action for action, q_value in self.q_table[state].items() if q_value == max_q_value]

            if len(best_actions) > 1:  # if there are ties
                # break ties by rolling a dice
                action = random.choice(best_actions)
            else:
                action = best_actions[0]  # choose the only action if there are no ties

            return action    
        
         
    def PExploit(self, state, actions): #exploits with a probability of 80% to choose from the action with the highest Q-value, otherwise choose an action at random

        if random.uniform(0, 1) < 0.80:
            max_q_value = max(self.q_table[state].values())
            best_actions = [action for action, q_value in self.q_table[state].items() if q_value == max_q_value]
            
            return random.choice(best_actions)
        else:
            return random.choice(actions)
        
    def update_q_value(self, state, action, reward, next_state, next_action):  #updates Q-value for a state-action pair based on observed reward and estimated future rewards
        if state not in self.q_table:
            self.q_table[state] = {}
        if next_state not in self.q_table:
            self.q_table[next_state] = {}

        if action not in self.q_table[state]:
            self.q_table[state][action] = 0
        if next_action not in self.q_table[next_state]:
            self.q_table[next_state][next_action] = 0

        # SARSA update rule
        self.q_table[state][action] += self.alpha * (
                    reward + self.gamma * self.q_table[next_state][next_action] - self.q_table[state][action])

###########################################          

def reset_board(agents, pickup_locations): #resets the board by updating positions, score, and block holdings of agents
    # Reset the board
    initial_positions = [(2, 2), (4, 2), (0, 2)]

    pickupLocations.clear()
    pickupLocations.update(pickup_locations)
    dropoffLocations.clear()
    dropoffLocations.update({(0, 0): 0, (2, 0): 0, (3, 4): 0})
    for i, agent in enumerate(agents):
        agent.position = initial_positions[i]
        agent.score = 0
        agent.have_block = False
    print("Board reset.")


def print_to_file(file_name, *args): #prints given agruments to output file
    with open(file_name, 'a') as file:
        print(*args, file=file)

def print_world(agents, output_file): #prints the current state of the world to output file
    world = np.zeros((world_size, world_size), dtype=str)
    for agent in agents:
        world[agent.position[0], agent.position[1]] = agent.name[0]

    print_to_file(output_file, "Current world state:")
    print_to_file(output_file, world)

def print_q_table(q_table, output_file): # prints Q-table to output file
    print_to_file(output_file, "Q-table:")
    for state, actions in q_table.items():
        print_to_file(output_file, "State:", state)
        action_values = []
        for action, value in actions.items():
            action_values.append((action, value))
        print_to_file(output_file, action_values)


def print_score(agents, output_file): #prints schores of each agent to output file
    print_to_file(output_file, "Scores:")
    for agent in agents:
        print_to_file(output_file, f"{agent.name}: {agent.score}")


###########################################  

# the next functions set up each experiment

def experiment_1A(): 
    agents = [Agent("Red", (2, 2)),
              Agent("Blue", (4, 2)), 
              Agent("Black", (0, 2))]

    q_agent = QLearningAgent(alpha=0.3, gamma=0.5)
    
    # Open a text file for writing
    output_file = "experiment_results.txt"
    with open(output_file, 'w') as file:
        file.write("Experiment 1A\n\n")

    # Print initial world state to the text file
    print_to_file(output_file, "Initial world state:")
    print_world(agents, output_file)

    operator_applications_limit = 9000
    operator_applications = 0
    terminal_state_reached = 0
    steps_for_previous_terminal = 0

    while operator_applications < 500:
        operator_applications += 1
        for agent in agents:
            actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # North, South, East, West
            action = q_agent.PRandom(agent.position, actions)
            if agent.move(action, agents):
                q_agent.update_q_value(agent.position, action, agent.reward, agent.position, action)
        if all(value == 5 for value in dropoffLocations.values()):
            terminal_state_reached += 1            
            print_to_file(output_file, f"Termination #{terminal_state_reached} condition met. All dropoff locations are full. Resetting Board\n")
            print_to_file(output_file, f"Steps taken to terminal state: {operator_applications-steps_for_previous_terminal}\n")
            steps_for_previous_terminal = operator_applications
            print_to_file(output_file, f"Scores after operator applications: {operator_applications}")
            for agent in agents:
                print_to_file(output_file, f"{agent.name}: {agent.score}")

            print_to_file(output_file, "World state after operator applications:")
            print_world(agents, output_file)
            print_to_file(output_file, "Q-table after operator applications:")
            print_q_table(q_agent.q_table, output_file)
            print_to_file(output_file, "---------------------------\n")

            pickupLocations = {(0, 4): 5, (1, 3): 5, (4, 1): 5}
            reset_board(agents, pickupLocations)

            # Check if Q-table remains unchanged after resetting the board
            # print_to_file(output_file, "Checking Q-table after board reset:")
            # print_q_table(q_agent.q_table, output_file)
            # print_to_file(output_file, "Q-table check complete.\n")
        if operator_applications == 500:
            print_to_file(output_file, "Initial 500 steps completed.\n")


    while operator_applications < operator_applications_limit:
        operator_applications += 1
        for agent in agents:
            actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # North, South, East, West
            action = q_agent.PRandom(agent.position, actions)
            if agent.move(action, agents):
                q_agent.update_q_value(agent.position, action, agent.reward, agent.position, action)

        if operator_applications % 500 == 0:
            print_to_file(output_file, "LET'S SHOW PROGRESS. CURRENT WORLD AND Q-TABLE.")
            print_to_file(output_file, f"Current operations applied: {operator_applications}")
            print_world(agents, output_file)
            print_q_table(q_agent.q_table, output_file)
    
        if all(value == 5 for value in dropoffLocations.values()):
            terminal_state_reached += 1
            print_to_file(output_file, f"Termination #{terminal_state_reached} condition met. All dropoff locations are full. Resetting Board\n")
            print_to_file(output_file, f"Steps taken to terminal state: {operator_applications-steps_for_previous_terminal}\n")
            steps_for_previous_terminal = operator_applications
            print_to_file(output_file, f"Scores after operator applications: {operator_applications}")
            for agent in agents:
                print_to_file(output_file, f"{agent.name}: {agent.score}")

            print_to_file(output_file, "World state after operator applications:")
            print_world(agents, output_file)
            print_to_file(output_file, "Q-table after operator applications:")
            print_q_table(q_agent.q_table, output_file)
            print_to_file(output_file, "---------------------------\n")

            # Resetting the board
            pickupLocations = {(0, 4): 5, (1, 3): 5, (4, 1): 5}
            reset_board(agents, pickupLocations)

            # Check if Q-table remains unchanged after resetting the board
            # print_to_file(output_file, "Checking Q-table after board reset:")
            # print_q_table(q_agent.q_table, output_file)
            # print_to_file(output_file, "Q-table check complete.\n")

    print_to_file(output_file, "Program has completed all operations.")
    print_to_file(output_file, f"Total number of Terminal States reached: {terminal_state_reached}")

def experiment_1B():
    agents = [Agent("Red", (2, 2)),
              Agent("Blue", (4, 2)), 
              Agent("Black", (0, 2))]
    
    q_agent = QLearningAgent(alpha=0.3, gamma=0.5)
    
    # Open a text file for writing
    output_file = "experiment_results.txt"
    with open(output_file, 'w') as file:
        file.write("Experiment 1B\n\n")

    # Print initial world state to the text file
    print_to_file(output_file, "Initial world state:")
    print_world(agents, output_file)

    operator_applications_limit = 9000
    operator_applications = 0
    terminal_state_reached = 0
    steps_for_previous_terminal = 0

    while operator_applications < 500:
        operator_applications += 1
        for agent in agents:
            actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # North, South, East, West
            action = q_agent.PRandom(agent.position, actions)
            if agent.move(action, agents):
                q_agent.update_q_value(agent.position, action, agent.reward, agent.position, action)
        if all(value == 5 for value in dropoffLocations.values()):
            terminal_state_reached += 1
            print_to_file(output_file, f"Termination #{terminal_state_reached} condition met. All dropoff locations are full. Resetting Board\n")
            print_to_file(output_file, f"Steps taken to terminal state: {operator_applications-steps_for_previous_terminal}\n")
            steps_for_previous_terminal = operator_applications
            print_to_file(output_file, f"Scores after operator applications: {operator_applications}")
            for agent in agents:
                print_to_file(output_file, f"{agent.name}: {agent.score}")

            print_to_file(output_file, "World state after first 500 operations")
            print_world(agents, output_file)
            print_to_file(output_file, "Q-table after operator applications:")
            print_q_table(q_agent.q_table, output_file)
            print_to_file(output_file, "---------------------------\n")

            # Resetting the board
            pickupLocations = {(0, 4): 5, (1, 3): 5, (4, 1): 5}
            reset_board(agents, pickupLocations)

            # Check if Q-table remains unchanged after resetting the board
            # print_to_file(output_file, "Checking Q-table after board reset:")
            # print_q_table(q_agent.q_table, output_file)
            # print_to_file(output_file, "Q-table check complete.\n")

    print_to_file(output_file, "Initial 500 steps completed.\n")

    while operator_applications < operator_applications_limit:
        operator_applications += 1
        for agent in agents:
            actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # North, South, East, West
            action = q_agent.PGreedy(agent.position, actions)
            if agent.move(action, agents):
                q_agent.update_q_value(agent.position, action, agent.reward, agent.position, action)

        if operator_applications % 500 == 0:
            print_to_file(output_file, "LET'S SHOW PROGRESS. CURRENT WORLD AND Q-TABLE.")
            print_to_file(output_file, f"Current operations applied: {operator_applications}")
            print_world(agents, output_file)
            print_q_table(q_agent.q_table, output_file)
            
        if all(value == 5 for value in dropoffLocations.values()):
            terminal_state_reached += 1
            print_to_file(output_file, f"Termination #{terminal_state_reached} condition met. All dropoff locations are full. Resetting Board\n")
            print_to_file(output_file, f"Steps taken to terminal state: {operator_applications-steps_for_previous_terminal}\n")
            steps_for_previous_terminal = operator_applications
            print_to_file(output_file, f"Scores after operator applications: {operator_applications}")
            for agent in agents:
                print_to_file(output_file, f"{agent.name}: {agent.score}")

            print_to_file(output_file, "World state after operator applications:")
            print_world(agents, output_file)
            print_to_file(output_file, "Q-table after operator applications:")
            print_q_table(q_agent.q_table, output_file)
            print_to_file(output_file, "---------------------------\n")

           # Resetting the board
            pickupLocations = {(0, 4): 5, (1, 3): 5, (4, 1): 5}
            reset_board(agents, pickupLocations)

            # Check if Q-table remains unchanged after resetting the board
            # print_to_file(output_file, "Checking Q-table after board reset:")
            # print_q_table(q_agent.q_table, output_file)
            # print_to_file(output_file, "Q-table check complete.\n")
    print_to_file(output_file, "Program has completed all operations.")
    print_to_file(output_file, f"Total number of Terminal States reached: {terminal_state_reached}")

def experiment_1C():
    agents = [Agent("Red", (2, 2)),
              Agent("Blue", (4, 2)), 
              Agent("Black", (0, 2))]
    
    q_agent = QLearningAgent(alpha=0.3, gamma=0.5)
    
    # Open a text file for writing
    output_file = "experiment_results.txt"
    with open(output_file, 'w') as file:
        file.write("Experiment 1C\n\n")

    # Print initial world state to the text file
    print_to_file(output_file, "Initial world state:")
    print_world(agents, output_file)

    operator_applications_limit = 9000
    operator_applications = 0
    terminal_state_reached = 0
    steps_for_previous_terminal = 0

    while operator_applications < 500:
        operator_applications += 1
        for agent in agents:
            actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # North, South, East, West
            action = q_agent.PRandom(agent.position, actions)
            if agent.move(action, agents):
                q_agent.update_q_value(agent.position, action, agent.reward, agent.position, action)
        if all(value == 5 for value in dropoffLocations.values()):
            terminal_state_reached += 1
            print_to_file(output_file, f"Termination #{terminal_state_reached} condition met. All dropoff locations are full. Resetting Board\n")
            print_to_file(output_file, f"Steps taken to terminal state: {operator_applications-steps_for_previous_terminal}\n")
            steps_for_previous_terminal = operator_applications
            print_to_file(output_file, f"Scores after operator applications: {operator_applications}")
            for agent in agents:
                print_to_file(output_file, f"{agent.name}: {agent.score}")

            print_to_file(output_file, "World state after operator applications:")
            print_world(agents, output_file)
            print_to_file(output_file, "Q-table after operator applications:")
            print_q_table(q_agent.q_table, output_file)
            print_to_file(output_file, "---------------------------\n")

            # Resetting the board
            pickupLocations = {(0, 4): 5, (1, 3): 5, (4, 1): 5}
            reset_board(agents, pickupLocations)

            # Check if Q-table remains unchanged after resetting the board
            # print_to_file(output_file, "Checking Q-table after board reset:")
            # print_q_table(q_agent.q_table, output_file)
            # print_to_file(output_file, "Q-table check complete.\n")

    print_to_file(output_file, "Initial 500 steps completed.\n")

    while operator_applications < operator_applications_limit:
        operator_applications += 1
        for agent in agents:
            actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # North, South, East, West
            action = q_agent.PExploit(agent.position, actions)
            if agent.move(action, agents):
                q_agent.update_q_value(agent.position, action, agent.reward, agent.position, action)
        
        if operator_applications % 500 == 0:
            print_to_file(output_file, "LET'S SHOW PROGRESS. CURRENT WORLD AND Q-TABLE.")
            print_to_file(output_file, f"Current operations applied: {operator_applications}")
            print_world(agents, output_file)
            print_q_table(q_agent.q_table, output_file)

        if all(value == 5 for value in dropoffLocations.values()):
            terminal_state_reached += 1
            print_to_file(output_file, f"Termination #{terminal_state_reached} condition met. All dropoff locations are full. Resetting Board\n")
            print_to_file(output_file, f"Steps taken to terminal state: {operator_applications-steps_for_previous_terminal}\n")
            steps_for_previous_terminal = operator_applications
            print_to_file(output_file, f"Scores after operator applications: {operator_applications}")
            for agent in agents:
                print_to_file(output_file, f"{agent.name}: {agent.score}")

            print_to_file(output_file, "World state after operator applications:")
            print_world(agents, output_file)
            print_to_file(output_file, "Q-table after operator applications:")
            print_q_table(q_agent.q_table, output_file)
            print_to_file(output_file, "---------------------------\n")

            # Resetting the board
            pickupLocations = {(0, 4): 5, (1, 3): 5, (4, 1): 5}
            reset_board(agents, pickupLocations)

            # Check if Q-table remains unchanged after resetting the board
            # print_to_file(output_file, "Checking Q-table after board reset:")
            # print_q_table(q_agent.q_table, output_file)
            # print_to_file(output_file, "Q-table check complete.\n")
    print_to_file(output_file, "Program has completed all operations.")
    print_to_file(output_file, f"Total number of Terminal States reached: {terminal_state_reached}")

def experiment_2():
    agents = [Agent("Red", (2, 2)),
              Agent("Blue", (4, 2)), 
              Agent("Black", (0, 2))]
    
    s_agent = SARSA(alpha=0.3, gamma=0.5)

    output_file = "experiment_results.txt"
    with open(output_file, 'w') as file:
        file.write("Experiment 2\n\n")


    print_to_file(output_file, "Initial world state:")
    print_world(agents, output_file)

    total_score = 0
    operator_applications_limit = 9000
    operator_applications = 0
    terminal_state_reached = 0
    steps_for_previous_terminal = 0

    # Initial 500 steps using PRandom policy
    while operator_applications < 500:
        operator_applications += 1
        for agent in agents:
            actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # North, South, East, West
            action = s_agent.PRandom(agent.position, actions)
            if agent.move(action, agents):
                s_agent.update_q_value(agent.position, action, agent.reward, agent.position, action)
                total_score += agent.score
        if all(value == 5 for value in dropoffLocations.values()):
            terminal_state_reached += 1
            print_to_file(output_file, f"Termination #{terminal_state_reached} condition met. All dropoff locations are full. Resetting Board\n")
            print_to_file(output_file, f"Steps taken to terminal state: {operator_applications-steps_for_previous_terminal}\n")
            steps_for_previous_terminal = operator_applications
            print_to_file(output_file, f"Scores after operator applications: {operator_applications}")
            for agent in agents:
                print_to_file(output_file, f"{agent.name}: {agent.score}")

            print_to_file(output_file, "World state after operator applications:")
            print_world(agents, output_file)
            print_to_file(output_file, "Q-table after operator applications:")
            print_q_table(s_agent.q_table, output_file)
            print_to_file(output_file, "---------------------------\n")

            # Resetting the board
            pickupLocations = {(0, 4): 5, (1, 3): 5, (4, 1): 5}
            reset_board(agents, pickupLocations)

    # Continue running PExploit for 8500 more steps
    while operator_applications < operator_applications_limit:
        operator_applications += 1
        for agent in agents:
            actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # North, South, East, West
            action = s_agent.PExploit(agent.position, actions)
            if agent.move(action, agents):
                # Perform pickup if applicable
                s_agent.update_q_value(agent.position, action, agent.reward, agent.position, action)
                total_score += agent.score
        
        if operator_applications % 500 == 0:
            print_to_file(output_file, "LET'S SHOW PROGRESS. CURRENT WORLD AND Q-TABLE.")
            print_to_file(output_file, f"Current operations applied: {operator_applications}")
            print_world(agents, output_file)
            print_q_table(s_agent.q_table, output_file)

        if all(value == 5 for value in dropoffLocations.values()):
            terminal_state_reached += 1
            print_to_file(output_file, f"Termination #{terminal_state_reached} condition met. All dropoff locations are full. Resetting Board\n")
            print_to_file(output_file, f"Steps taken to terminal state: {operator_applications-steps_for_previous_terminal}\n")
            steps_for_previous_terminal = operator_applications
            print_to_file(output_file, f"Scores after operator applications: {operator_applications}")
            for agent in agents:
                print_to_file(output_file, f"{agent.name}: {agent.score}")

            print_to_file(output_file, "World state after operator applications:")
            print_world(agents, output_file)
            print_to_file(output_file, "Q-table after operator applications:")
            print_q_table(s_agent.q_table, output_file)
            print_to_file(output_file, "---------------------------\n")

            # Resetting the board
            pickupLocations = {(0, 4): 5, (1, 3): 5, (4, 1): 5}
            reset_board(agents, pickupLocations)
    print_to_file(output_file, "Program has completed all operations.")
    print_to_file(output_file, f"Total number of Terminal States reached: {terminal_state_reached}")

def experiment_3A():
    agents = [Agent("Red", (2, 2)),
              Agent("Blue", (4, 2)), 
              Agent("Black", (0, 2))]
    
    s_agent = SARSA(alpha=0.15, gamma=0.5)

    output_file = "experiment_results.txt"
    with open(output_file, 'w') as file:
        file.write("Experiment 3A\n\n")


    print_to_file(output_file, "Initial world state:")
    print_world(agents, output_file)

    total_score = 0
    operator_applications_limit = 9000
    operator_applications = 0
    terminal_state_reached = 0
    steps_for_previous_terminal = 0

    # Initial 500 steps using PRandom policy
    while operator_applications < 500:
        operator_applications += 1
        for agent in agents:
            actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # North, South, East, West
            action = s_agent.PRandom(agent.position, actions)
            if agent.move(action, agents):
                s_agent.update_q_value(agent.position, action, agent.reward, agent.position, action)
                total_score += agent.score
        if all(value == 5 for value in dropoffLocations.values()):
            terminal_state_reached += 1
            print_to_file(output_file, f"Termination #{terminal_state_reached} condition met. All dropoff locations are full. Resetting Board\n")
            print_to_file(output_file, f"Steps taken to terminal state: {operator_applications-steps_for_previous_terminal}\n")
            steps_for_previous_terminal = operator_applications
            print_to_file(output_file, f"Scores after operator applications: {operator_applications}")
            for agent in agents:
                print_to_file(output_file, f"{agent.name}: {agent.score}")

            print_to_file(output_file, "World state after operator applications:")
            print_world(agents, output_file)
            print_to_file(output_file, "Q-table after operator applications:")
            print_q_table(s_agent.q_table, output_file)
            print_to_file(output_file, "---------------------------\n")

            # Resetting the board
            pickupLocations = {(0, 4): 5, (1, 3): 5, (4, 1): 5}
            reset_board(agents, pickupLocations)

    # Continue running PExploit for 8500 more steps
    while operator_applications < operator_applications_limit:
        operator_applications += 1
        for agent in agents:
            actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # North, South, East, West
            action = s_agent.PExploit(agent.position, actions)
            if agent.move(action, agents):
                # Perform pickup if applicable
                s_agent.update_q_value(agent.position, action, agent.reward, agent.position, action)
                total_score += agent.score

        if operator_applications % 500 == 0:
            print_to_file(output_file, "LET'S SHOW PROGRESS. CURRENT WORLD AND Q-TABLE.")
            print_to_file(output_file, f"Current operations applied: {operator_applications}")
            print_world(agents, output_file)
            print_q_table(s_agent.q_table, output_file)

        if all(value == 5 for value in dropoffLocations.values()):
            terminal_state_reached += 1
            print_to_file(output_file, f"Termination #{terminal_state_reached} condition met. All dropoff locations are full. Resetting Board\n")
            print_to_file(output_file, f"Steps taken to terminal state: {operator_applications-steps_for_previous_terminal}\n")
            steps_for_previous_terminal = operator_applications
            print_to_file(output_file, f"Scores after operator applications: {operator_applications}")
            for agent in agents:
                print_to_file(output_file, f"{agent.name}: {agent.score}")

            print_to_file(output_file, "World state after operator applications:")
            print_world(agents, output_file)
            print_to_file(output_file, "Q-table after operator applications:")
            print_q_table(s_agent.q_table, output_file)
            print_to_file(output_file, "---------------------------\n")

            # Resetting the board
            pickupLocations = {(0, 4): 5, (1, 3): 5, (4, 1): 5}
            reset_board(agents, pickupLocations)
    print_to_file(output_file, "Program has completed all operations.")
    print_to_file(output_file, f"Total number of Terminal States reached: {terminal_state_reached}")

def experiment_3B():
    agents = [Agent("Red", (2, 2)),
              Agent("Blue", (4, 2)), 
              Agent("Black", (0, 2))]
    
    s_agent = SARSA(alpha=0.45, gamma=0.5)

    output_file = "experiment_results.txt"
    with open(output_file, 'w') as file:
        file.write("Experiment 3B\n\n")


    print_to_file(output_file, "Initial world state:")
    print_world(agents, output_file)

    total_score = 0
    operator_applications_limit = 9000
    operator_applications = 0
    terminal_state_reached = 0
    steps_for_previous_terminal = 0

    # Initial 500 steps using PRandom policy
    while operator_applications < 500:
        operator_applications += 1
        for agent in agents:
            actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # North, South, East, West
            action = s_agent.PRandom(agent.position, actions)
            if agent.move(action, agents):
                s_agent.update_q_value(agent.position, action, agent.reward, agent.position, action)
                total_score += agent.score
        if all(value == 5 for value in dropoffLocations.values()):
            terminal_state_reached += 1
            print_to_file(output_file, f"Termination #{terminal_state_reached} condition met. All dropoff locations are full. Resetting Board\n")
            print_to_file(output_file, f"Steps taken to terminal state: {operator_applications-steps_for_previous_terminal}\n")
            steps_for_previous_terminal = operator_applications
            print_to_file(output_file, f"Scores after operator applications: {operator_applications}")
            for agent in agents:
                print_to_file(output_file, f"{agent.name}: {agent.score}")

            print_to_file(output_file, "World state after operator applications:")
            print_world(agents, output_file)
            print_to_file(output_file, "Q-table after operator applications:")
            print_q_table(s_agent.q_table, output_file)
            print_to_file(output_file, "---------------------------\n")

            # Resetting the board
            pickupLocations = {(0, 4): 5, (1, 3): 5, (4, 1): 5}
            reset_board(agents, pickupLocations)


    # Continue running PExploit for 8500 more steps
    while operator_applications < operator_applications_limit:
        operator_applications += 1
        for agent in agents:
            actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # North, South, East, West
            action = s_agent.PExploit(agent.position, actions)
            if agent.move(action, agents):
                # Perform pickup if applicable
                s_agent.update_q_value(agent.position, action, agent.reward, agent.position, action)
                total_score += agent.score

        if operator_applications % 500 == 0:
            print_to_file(output_file, "LET'S SHOW PROGRESS. CURRENT WORLD AND Q-TABLE.")
            print_to_file(output_file, f"Current operations applied: {operator_applications}")
            print_world(agents, output_file)
            print_q_table(s_agent.q_table, output_file)

        if all(value == 5 for value in dropoffLocations.values()):
            terminal_state_reached += 1
            print_to_file(output_file, f"Termination #{terminal_state_reached} condition met. All dropoff locations are full. Resetting Board\n")
            print_to_file(output_file, f"Steps taken to terminal state: {operator_applications-steps_for_previous_terminal}\n")
            steps_for_previous_terminal = operator_applications
            print_to_file(output_file, f"Scores after operator applications: {operator_applications}")
            for agent in agents:
                print_to_file(output_file, f"{agent.name}: {agent.score}")

            print_to_file(output_file, "World state after operator applications:")
            print_world(agents, output_file)
            print_to_file(output_file, "Q-table after operator applications:")
            print_q_table(s_agent.q_table, output_file)
            print_to_file(output_file, "---------------------------\n")

            # Resetting the board
            pickupLocations = {(0, 4): 5, (1, 3): 5, (4, 1): 5}
            reset_board(agents, pickupLocations)
    print_to_file(output_file, "Program has completed all operations.")
    print_to_file(output_file, f"Total number of Terminal States reached: {terminal_state_reached}")


def experiment_4():

    agents = [Agent("Red", (2, 2)),
              Agent("Blue", (4, 2)), 
              Agent("Black", (0, 2))]
    
    s_agent = SARSA(alpha=0.3, gamma=0.5)

    output_file = "experiment_results.txt"
    with open(output_file, 'w') as file:
        file.write("Experiment 4\n\n")


    print_to_file(output_file, "Initial world state:")
    print_world(agents, output_file)

    total_score = 0
    operator_applications = 0
    terminal_state_reached = 0
    steps_for_previous_terminal = 0

    # Initial 500 steps using PRandom policy
    while operator_applications < 500 :
        operator_applications += 1
        for agent in agents:
            actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # North, South, East, West
            action = s_agent.PRandom(agent.position, actions)
            if agent.move(action, agents):
                s_agent.update_q_value(agent.position, action, agent.reward, agent.position, action)
                total_score += agent.score
        
        if terminal_state_reached < 3:
            if all(value == 5 for value in dropoffLocations.values()):
                terminal_state_reached += 1
                print_to_file(output_file, f"Termination #{terminal_state_reached} condition met. All dropoff locations are full. Resetting Board\n")
                print_to_file(output_file, f"Steps taken to terminal state: {operator_applications-steps_for_previous_terminal}\n")
                steps_for_previous_terminal = operator_applications
                print_to_file(output_file, f"Scores after operator applications: {operator_applications}")
                for agent in agents:
                    print_to_file(output_file, f"{agent.name}: {agent.score}")

                print_to_file(output_file, "World state after operator applications:")
                print_world(agents, output_file)
                print_to_file(output_file, "Q-table after operator applications:")
                print_q_table(s_agent.q_table, output_file)
                print_to_file(output_file, "---------------------------\n")

                # Resetting the board
                pickupLocations = {(0, 4): 5, (1, 3): 5, (4, 1): 5}
                reset_board(agents, pickupLocations)
        
        if terminal_state_reached == 3:
            print_to_file(output_file, "3 Terminal States Reached. \n")
            if all(value == 5 for value in dropoffLocations.values()):
                terminal_state_reached += 1
                print_to_file(output_file, f"Termination #{terminal_state_reached} condition met. All dropoff locations are full. Resetting Board\n")
                print_to_file(output_file, f"Steps taken to terminal state: {operator_applications-steps_for_previous_terminal}\n")
                steps_for_previous_terminal = operator_applications
                print_to_file(output_file, f"Scores after operator applications: {operator_applications}")
                for agent in agents:
                    print_to_file(output_file, f"{agent.name}: {agent.score}")

                print_to_file(output_file, "World state after operator applications:")
                print_world(agents, output_file)
                print_to_file(output_file, "Q-table after operator applications:")
                print_q_table(s_agent.q_table, output_file)
                print_to_file(output_file, "---------------------------\n")

                # Resetting the board
                NewpickupLocations = {(3, 1): 5, (2, 2): 5, (1, 3): 5}
                reset_board(agents, NewpickupLocations)


    while terminal_state_reached < 6:
        operator_applications += 1
        for agent in agents:
            actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # North, South, East, West
            action = s_agent.PExploit(agent.position, actions)
            if agent.move(action, agents):
                # Perform pickup if applicable
                s_agent.update_q_value(agent.position, action, agent.reward, agent.position, action)
                total_score += agent.score

        if operator_applications % 500 == 0:
            print_to_file(output_file, "LET'S SHOW PROGRESS. CURRENT WORLD AND Q-TABLE.")
            print_to_file(output_file, f"Current operations applied: {operator_applications}")
            print_world(agents, output_file)
            print_q_table(s_agent.q_table, output_file)
            
        if all(value == 5 for value in dropoffLocations.values()):
            terminal_state_reached += 1
            print_to_file(output_file, f"Termination #{terminal_state_reached} condition met. All dropoff locations are full. Resetting Board\n")
            print_to_file(output_file, f"Steps taken to terminal state: {operator_applications-steps_for_previous_terminal}\n")
            steps_for_previous_terminal = operator_applications
            print_to_file(output_file, f"Scores after operator applications: {operator_applications}")
            for agent in agents:
                print_to_file(output_file, f"{agent.name}: {agent.score}")

            print_to_file(output_file, "World state after operator applications:")
            print_world(agents, output_file)
            print_to_file(output_file, "Q-table after operator applications:")
            print_q_table(s_agent.q_table, output_file)
            print_to_file(output_file, "---------------------------\n")

            # Resetting the board
            NewpickupLocations = {(3, 1): 5, (2, 2): 5, (1, 3): 5}
            reset_board(agents, NewpickupLocations)

    print_to_file(output_file, "Program has completed all operations.")
    print_to_file(output_file, f"Total number of Terminal States reached: {terminal_state_reached}")

###########################################


def main(): #menu that allows us to choose which experiment to be run using the functions previously stated
    print("[ Experiments Menu: 1A, 1B, 1C, 2, 3A, 3B, 4 ]")
    experiment = input('Enter which experiment you want to run: ')
    
    if experiment == "1A":
        experiment_1A()
    elif experiment == "1B":
        experiment_1B()
    elif experiment == "1C":
        experiment_1C()
    elif experiment == "2":
        experiment_2()
    elif experiment == "3A":
        experiment_3A()
    elif experiment == "3B":
        experiment_3B()
    elif experiment == "4":
        experiment_4()

if __name__ == "__main__":
    main()
