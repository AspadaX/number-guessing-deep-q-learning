import json
import os
import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn
from tqdm import tqdm

# Set device to CPU or GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define an auto-naming program
class NamingComponent:
    
    def __init__(self, model_class: str) -> None:
        self.__model_class: str = model_class
        self.__iteration: int = self.__load_iteration()
    
    def __load_iteration(self) -> int:
        """
        get the previous iteration by looking over the existing files
        """
        files: list[str] = os.listdir("./")
        max_iteration: int = 0

        for file in files:
            if file.endswith(".pt") and file.startswith(self.__model_class):
                try:
                    iteration = int(file.split("_")[1].split(".")[0])
                    if iteration > max_iteration:
                        max_iteration = iteration
                except ValueError:
                    pass

        return max_iteration
    
    def generate(self) -> str:
        return self.__model_class + "_" + str(self.__iteration + 1) + ".pt"
    
    def generate_graph_name(self) -> str:
        return self.__model_class + "_" + str(self.__iteration + 1) + ".png"
    
    def get_existing_iteration_model_name(self) -> str:
        return self.__model_class + "_" + str(self.__iteration) + ".pt"

# Define model
class DeepQLearning(nn.Module):
    def __init__(self, in_states: int, h1_nodes: int, out_actions: int):
        super().__init__()

        # Define network layers
        self.fc1 = nn.Linear(in_states, h1_nodes)   # first fully connected layer
        self.fc2 = nn.Linear(h1_nodes, h1_nodes)    # second fully connected layer
        self.out = nn.Linear(h1_nodes, out_actions) # output layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))  # Apply rectified linear unit (ReLU) activation
        x = F.relu(self.fc2(x))
        x = self.out(x)          # Calculate output
        return x


# Define memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen: int) -> None:
        self.memory: deque = deque([], maxlen=maxlen)

    def append(self, transition: tuple) -> None:
        self.memory.append(transition)

    def sample(self, sample_size: int) -> list:
        return random.sample(self.memory, sample_size)

    def __len__(self) -> int:
        return len(self.memory)


# Handle the training parameters
class TrainingParameters:
    
    def __init__(
        self, 
        learning_rate_a: float = 0.0001, 
        discount_factor_g: float = 0.9, 
        network_sync_rate: int = 50, 
        replay_memory_size: int = 10000, 
        mini_batch_size: int = 128
    ) -> None:
        # Hyperparameters (adjustable)
        self.learning_rate_a = learning_rate_a        # Learning rate
        self.discount_factor_g = discount_factor_g    # Discount factor (gamma)
        self.network_sync_rate = network_sync_rate    # Number of steps before syncing target network
        self.replay_memory_size = replay_memory_size  # Replay memory size
        self.mini_batch_size = mini_batch_size        # Mini-batch size
    
    def save_parameters(self, path: str) -> None:
        parameters: dict = {
            "learning_rate_a": self.learning_rate_a,
            "discount_factor_g": self.discount_factor_g,
            "network_sync_rate": self.network_sync_rate,
            "replay_memory_size": self.replay_memory_size,
            "mini_batch_size": self.mini_batch_size
        }

        if not path.endswith(".json"):
            path = path + ".json"
       
        with open(path, 'w') as file:
            json.dump(parameters, file, indent=4)

# Number Guessing Game Deep Q-Learning
class NumberGuessingGameDeepQLearning:

    # Neural Network
    loss_fn: nn.Module = nn.MSELoss()       # NN Loss function
    optimizer: torch.optim.Optimizer = None # NN Optimizer. Initialize later

    def __init__(
        self, 
        training_parameters: TrainingParameters, 
        min_value: int = 1, 
        max_value: int = 100, 
        max_attempts: int = 10
    ) -> None:
        self.target_number: int = 0
        self.__naming_component = NamingComponent(__file__.split(".")[0].split("/")[-1])
        self.__model_name = self.__naming_component.generate()
        self.__png_name = self.__naming_component.generate_graph_name()

        # Use hyperparameters from training parameters
        self.learning_rate_a: float = training_parameters.learning_rate_a
        self.discount_factor_g: float = training_parameters.discount_factor_g
        self.network_sync_rate: int = training_parameters.network_sync_rate
        self.replay_memory_size: int = training_parameters.replay_memory_size
        self.mini_batch_size: int = training_parameters.mini_batch_size
        
        # Store the training parameters
        self.__training_parameters = training_parameters

        # Dynamic range and attempts
        self.min_value = min_value
        self.max_value = max_value
        self.max_attempts = max_attempts
        
        # Define action space
        # Actions: 0: Guess at 25%
        #          1: Guess at 50% (midpoint)
        #          2: Guess at 75%
        self.num_actions = 3
        self.action_fractions = [0.25, 0.5, 0.75]

        # Initialize turn_count
        self.turn_count = 0

    def get_model_name(self) -> str:
        return self.__model_name

    # Train the Number Guessing Game
    def train(self, episodes: int) -> None:

        state_size: int = 2  # low_bound and high_bound
        num_actions: int = self.num_actions

        epsilon: float = 1.0  # 1 = 100% random actions
        epsilon_decay = 1 / (episodes * 0.5)  # Decay epsilon over half the episodes
        memory: ReplayMemory = ReplayMemory(self.replay_memory_size)

        # Create policy and target networks
        policy_dqn: DeepQLearning = DeepQLearning(in_states=state_size, h1_nodes=64, out_actions=num_actions).to(device)
        target_dqn: DeepQLearning = DeepQLearning(in_states=state_size, h1_nodes=64, out_actions=num_actions).to(device)

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        logger.info('Policy (random, before training):')

        # Policy network optimizer
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        # List to keep track of rewards collected per episode
        rewards_per_episode: np.ndarray = np.zeros(episodes)

        # List to keep track of epsilon decay
        epsilon_history: list[float] = []

        # Track number of steps taken (used for syncing policy => target network)
        step_count: int = 0

        for i in tqdm(range(episodes), desc="Training Progress"):
            self.turn_count = 0  # Reset turn_count at the start of each episode
            self.target_number = random.randint(self.min_value, self.max_value)  # Randomly select a target number
            # Initialize state: low and high bounds normalized
            low_bound = self.min_value
            high_bound = self.max_value
            state_input = torch.tensor([low_bound, high_bound], dtype=torch.float32, device=device)
            terminated: bool = False

            while not terminated and self.turn_count < self.max_attempts:
                # Normalize state inputs
                state_input_normalized = self.normalize_state(state_input)

                # Select action based on epsilon-greedy policy
                if random.random() < epsilon:
                    # Select random action
                    action: int = random.randint(0, num_actions - 1)
                else:
                    # Select best action
                    with torch.no_grad():
                        q_values = policy_dqn(state_input_normalized)
                        action = q_values.argmax().item()

                # Execute action
                guess = self.action_to_guess(action, state_input[0].item(), state_input[1].item())
                reward, terminated, feedback = self.step(guess)

                # Save current state and action
                current_state_input = state_input.clone()

                # Update state input with the feedback received
                state_input = self.update_state_input(state_input, guess, feedback)

                # Save experience into memory
                memory.append((self.normalize_state(current_state_input), action, reward, self.normalize_state(state_input), terminated))

                # Increment counters
                step_count += 1

                # Keep track of the rewards collected per episode
                if reward > 0:
                    rewards_per_episode[i] = reward  # Record the positive reward
                    logger.debug(f"Episode {i+1}/{episodes} - Target number: {self.target_number} - Guessed number: {guess} - Reward: {reward}")

                # Check if enough experience has been collected
                if len(memory) > self.mini_batch_size:
                    mini_batch: list[tuple] = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    # Decay epsilon
                    epsilon = max(epsilon - epsilon_decay, 0.1)
                    epsilon_history.append(epsilon)

                    # Copy policy network to target network after a certain number of steps
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0

            # Ensure epsilon history is updated even if memory is not large enough
            if len(memory) <= self.mini_batch_size:
                epsilon_history.append(epsilon)

        # Save policy
        torch.save(policy_dqn.state_dict(), self.__model_name)

        # Create plots
        plt.figure(1)

        # Plot average rewards per episode
        sum_rewards: np.ndarray = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.mean(rewards_per_episode[max(0, x - 100):(x + 1)])
        plt.subplot(121)  # Plot on a 1 row x 2 col grid, at cell 1
        plt.plot(sum_rewards)
        plt.title('Average Rewards per Episode')
        plt.xlabel('Episodes')
        plt.ylabel('Average Rewards')

        # Plot epsilon decay per episode
        plt.subplot(122)  # Plot on a 1 row x 2 col grid, at cell 2
        plt.plot(epsilon_history)
        plt.title('Epsilon Decay per Episode')
        plt.xlabel('Episodes')
        plt.ylabel('Epsilon')

        # Save plots
        plt.savefig(self.__png_name)
        
        # Save training parameters
        self.__training_parameters.save_parameters(path=self.__model_name)

    # Optimize policy network
    def optimize(self, mini_batch: list[tuple], policy_dqn: DeepQLearning, target_dqn: DeepQLearning) -> None:
        # Unpack the minibatch and convert to tensors
        state_inputs = torch.stack([transition[0].to(device) for transition in mini_batch])
        actions = torch.tensor([transition[1] for transition in mini_batch], dtype=torch.long, device=device)
        rewards = torch.tensor([transition[2] for transition in mini_batch], dtype=torch.float32, device=device)
        next_state_inputs = torch.stack([transition[3].to(device) for transition in mini_batch])
        terminateds = torch.tensor([transition[4] for transition in mini_batch], dtype=torch.bool, device=device)

        # Compute Q(s, a) using the policy network
        current_q_values = policy_dqn(state_inputs).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q values
        with torch.no_grad():
            # Compute max Q(s', a') for the next states using the target network
            next_q_values = target_dqn(next_state_inputs)
            max_next_q_values = next_q_values.max(dim=1)[0]
            target_q_values = rewards + (self.discount_factor_g * max_next_q_values * (~terminateds))

        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    '''
    Updates the input tensor by adjusting the low and high bounds based on the feedback.

    Parameters:
    - state_input: torch.Tensor (current state)
    - guess: int (number guessed)
    - feedback: str ('higher', 'lower', or 'correct')

    Returns:
    - torch.Tensor (updated state)
    '''
    def update_state_input(
        self,
        state_input: torch.Tensor,
        guess: int,
        feedback: str
    ) -> torch.Tensor:
        low_bound = state_input[0]
        high_bound = state_input[1]

        if feedback == 'higher':
            # Update low_bound
            low_bound = max(low_bound, guess + 1)
        elif feedback == 'lower':
            # Update high_bound
            high_bound = min(high_bound, guess - 1)
        elif feedback == 'correct':
            # Guess is correct; bounds are now the guessed number
            low_bound = guess
            high_bound = guess

        # Ensure bounds are within valid range
        low_bound = max(low_bound, self.min_value)
        high_bound = min(high_bound, self.max_value)
        return torch.tensor([low_bound, high_bound], dtype=torch.float32, device=device)

    def step(self, guess: int) -> tuple[float, bool, str]:
        self.turn_count += 1  # Increment turn count

        if guess == self.target_number:
            # Reward inversely proportional to the number of steps taken
            reward = (self.max_attempts - self.turn_count + 1) / self.max_attempts
            return reward, True, 'correct'
        else:
            # Calculate the negative reward proportional to the number of turns taken
            reward = -1.0 / self.max_attempts
            # Provide feedback whether the guess should be higher or lower
            feedback = 'lower' if guess > self.target_number else 'higher'
            return reward, False, feedback

    def action_to_guess(self, action: int, low_bound: float, high_bound: float) -> int:
        fraction = self.action_fractions[action]
        guess = low_bound + fraction * (high_bound - low_bound)
        return int(np.round(guess))

    def normalize_state(self, state_input: torch.Tensor) -> torch.Tensor:
        # Normalize low and high bounds to [0,1]
        normalized_low = (state_input[0] - self.min_value) / (self.max_value - self.min_value)
        normalized_high = (state_input[1] - self.min_value) / (self.max_value - self.min_value)
        return torch.stack([normalized_low, normalized_high])

    # Run the Number Guessing Game with the learned policy and compare with binary search
    def test(self, episodes: int) -> None:
        state_size: int = 2
        num_actions: int = self.num_actions

        # Load learned policy
        policy_dqn: DeepQLearning = DeepQLearning(in_states=state_size, h1_nodes=64, out_actions=num_actions)
        policy_dqn.load_state_dict(torch.load(self.__naming_component.get_existing_iteration_model_name(), map_location=device))
        policy_dqn.to(device)
        policy_dqn.eval()    # Switch model to evaluation mode

        logger.info('Policy (trained):')

        # Stats for the trained model
        model_wins = 0  # Initialize win counter
        model_total_turns = 0  # Initialize total turns counter
        model_fastest_turn = float('inf')  # Initialize fastest turn counter
        model_slowest_turn = 0  # Initialize slowest turn counter
        model_turns_list = []

        # Stats for binary search
        bs_total_turns = 0  # Initialize total turns counter for binary search
        bs_fastest_turn = float('inf')  # Initialize fastest turn counter for binary search
        bs_slowest_turn = 0  # Initialize slowest turn counter for binary search
        bs_turns_list = []

        for i in range(episodes):
            # Generate a random target number
            target_number = random.randint(self.min_value, self.max_value)

            # ==== Test the Trained Model ====
            self.turn_count = 0  # Reset turn_count at the start of each episode
            logger.debug("==========================")
            self.target_number = target_number  # Set the target number
            logger.debug(f"Episode {i+1}/{episodes} - Target number: {self.target_number}")
            # Initialize state: low and high bounds
            low_bound = self.min_value
            high_bound = self.max_value
            state_input = torch.tensor([low_bound, high_bound], dtype=torch.float32, device=device)
            terminated: bool = False

            while not terminated and self.turn_count < self.max_attempts:
                with torch.no_grad():
                    # Normalize state inputs
                    state_input_normalized = self.normalize_state(state_input)
                    q_values = policy_dqn(state_input_normalized)
                    action = q_values.argmax().item()
                guess = self.action_to_guess(action, state_input[0].item(), state_input[1].item())
                logger.debug(f"Model - Selected action: {action}, Guess: {guess}")

                # Execute action
                reward, terminated, feedback = self.step(guess)
                logger.debug(f"Model - Reward: {reward}, Terminated: {terminated}, Feedback: {feedback}\n")

                # Update state input with the feedback received
                state_input = self.update_state_input(state_input, guess, feedback)

            if reward > 0:
                model_wins += 1
                model_total_turns += self.turn_count
                model_turns_list.append(self.turn_count)
                if self.turn_count < model_fastest_turn:
                    model_fastest_turn = self.turn_count
                if self.turn_count > model_slowest_turn:
                    model_slowest_turn = self.turn_count

            # ==== Perform Binary Search ====
            bs_turn_count = 0
            bs_low = self.min_value
            bs_high = self.max_value
            bs_found = False

            while bs_low <= bs_high:
                bs_turn_count += 1
                bs_guess = (bs_low + bs_high) // 2
                logger.debug(f"Binary Search - Guess: {bs_guess}")

                if bs_guess == target_number:
                    bs_found = True
                    break
                elif bs_guess < target_number:
                    bs_low = bs_guess + 1
                else:
                    bs_high = bs_guess - 1

            bs_total_turns += bs_turn_count
            bs_turns_list.append(bs_turn_count)
            if bs_turn_count < bs_fastest_turn:
                bs_fastest_turn = bs_turn_count
            if bs_turn_count > bs_slowest_turn:
                bs_slowest_turn = bs_turn_count

        # ==== Output Statistics ====
        # Trained Model Statistics
        model_win_rate = model_wins / episodes * 100
        if model_wins > 0:
            model_average_turns = model_total_turns / model_wins
        else:
            model_average_turns = float('inf')

        logger.info(f"===== Trained Model Statistics =====")
        logger.info(f"Win rate: {model_win_rate:.2f}%")
        if model_wins > 0:
            logger.info(f"Average turns needed to win: {model_average_turns:.2f}")
            logger.info(f"Fastest turn to win: {model_fastest_turn}")
            logger.info(f"Slowest turn to win: {model_slowest_turn}")
        else:
            logger.info("No wins recorded.")

        # Binary Search Statistics
        bs_average_turns = bs_total_turns / episodes

        logger.info(f"===== Binary Search Statistics =====")
        logger.info(f"Average turns needed: {bs_average_turns:.2f}")
        logger.info(f"Fastest turn: {bs_fastest_turn}")
        logger.info(f"Slowest turn: {bs_slowest_turn}")

        # Comparison
        logger.info(f"===== Comparison =====")
        if model_wins > 0:
            logger.info(f"Average turns (Model) vs (Binary Search): {model_average_turns:.2f} vs {bs_average_turns:.2f}")
        else:
            logger.info(f"Average turns (Model) vs (Binary Search): No wins vs {bs_average_turns:.2f}")

        # Plotting distribution of turns
        plt.figure(figsize=(12, 6))
        plt.hist(model_turns_list, bins=range(1, self.max_attempts + 2), alpha=0.5, label='Model')
        plt.hist(bs_turns_list, bins=range(1, self.max_attempts + 2), alpha=0.5, label='Binary Search')
        plt.title('Distribution of Turns Needed')
        plt.xlabel('Number of Turns')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig("comparison_" + self.__png_name)

    # Optionally, you can implement print_dqn to inspect the policy for all states
    def print_dqn(self, dqn: DeepQLearning) -> None:
        pass  # Implementation depends on how you want to represent and print the states


if __name__ == "__main__":

    import sys

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Specify dynamic range and attempts
    min_value = 1
    max_value = 100000
    max_attempts = int(np.ceil(np.log2(max_value - min_value + 1)) + 1)  # Slightly more than optimal binary search steps
    # max_attempts = 100

    instruction: str = input("Train or Test the Number Guessing Game? y/n (y to train, n to test): ").strip().lower()
    if instruction == 'y':
        # Train the Number Guessing Game
        number_guessing_game_deep_q_learning = NumberGuessingGameDeepQLearning(
            training_parameters=TrainingParameters(
                learning_rate_a=0.00001,  # Learning rate
                discount_factor_g=0.95,   # Discount factor (gamma)
                network_sync_rate=50,    # Network sync rate
                replay_memory_size=25000,  # Replay memory size
                mini_batch_size=512      # Mini-batch size
            ),
            min_value=min_value,
            max_value=max_value,
            max_attempts=max_attempts
        )
        logger.info(f"Model: {number_guessing_game_deep_q_learning.get_model_name()}")
        number_guessing_game_deep_q_learning.train(5000)
    elif instruction == 'n':
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
        # Test the Number Guessing Game
        number_guessing_game_deep_q_learning = NumberGuessingGameDeepQLearning(
            training_parameters=TrainingParameters(),
            min_value=min_value,
            max_value=max_value,
            max_attempts=max_attempts
        )
        logger.info(f"Model: {number_guessing_game_deep_q_learning.get_model_name()}")
        number_guessing_game_deep_q_learning.test(100)