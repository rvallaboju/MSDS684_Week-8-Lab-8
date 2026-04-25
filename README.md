# Comprehensive Algorithm Comparison on CartPole-v1

This repository contains the code for a comprehensive comparison of three distinct reinforcement learning (RL) algorithms on the classic CartPole-v1 environment. The goal is to evaluate and understand the trade-offs in sample efficiency, variance, and implementation complexity across different RL methodologies, ranging from tabular methods to function approximation techniques.

## Table of Contents

- [Project Overview](#project-overview)
- [Algorithms Implemented](#algorithms-implemented)
- [Key Findings](#key-findings)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The CartPole-v1 environment from Gymnasium is a widely used benchmark in reinforcement learning where an agent must balance a pole on a moving cart. This project systematically compares:

1.  **Tabular Q-Learning** (with state discretization)
2.  **SARSA with Tile Coding** (using linear function approximation)
3.  **REINFORCE with Baseline** (a policy gradient method with neural networks)

By comparing these algorithms, we aim to illustrate the evolution of RL techniques, highlighting how they address the challenges of continuous state spaces and complex policies.

## Algorithms Implemented

### 1. Tabular Q-Learning

This method discretizes the continuous state space of CartPole-v1 into a finite number of bins. A Q-table is then used to store Q-values for each state-action pair, updated using the Q-learning algorithm with epsilon-greedy exploration.

### 2. SARSA with Tile Coding

To handle the continuous state space more effectively, SARSA (State-Action-Reward-State-Action) is implemented with tile coding. This technique uses a fixed number of overlapping tilings to create binary features, allowing for linear function approximation of the Q-values. This provides a balance between generalization and granularity.

### 3. REINFORCE with Baseline

REINFORCE is a Monte Carlo policy gradient method. This implementation enhances REINFORCE by incorporating a state-value baseline (critic) to reduce variance in the policy gradient estimates. Both the policy (actor) and value (critic) networks are represented by simple feed-forward neural networks.

## Key Findings

-   **SARSA with Tile Coding** demonstrated the highest average final reward (214.00 ± 73.03) across multiple seeds, indicating a strong balance between generalization and learning stability. It provides a good trade-off in continuous state spaces.
-   **REINFORCE with Baseline** achieved an average final reward of (191.33 ± 219.54). While capable of achieving optimal performance (500 reward) in some runs, its high variance across different random seeds highlights its sensitivity to initialization and hyperparameter tuning.
-   **Tabular Q-Learning** (with 10 bins per dimension) consistently yielded a low final reward (10.00 ± 0.00). The coarse discretization leads to a significant loss of state information and struggles to generalize effectively in the continuous environment.

This project underscores the importance of function approximation techniques for continuous state spaces and demonstrates that well-designed linear features (like tile coding) can sometimes outperform more complex neural network-based methods in terms of average performance and stability.

## Setup and Installation

To run this project, you need Python 3.8+.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/rvallaboju/MSDS684_Week-8-Lab-8
    cd MSDS684_Week-8-Lab-8
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**

    The project relies on the following Python packages:

    *   `numpy`
    *   `matplotlib`
    *   `gymnasium`
    *   `torch`

    You can install them using `pip`:

    ```bash
    pip install numpy matplotlib gymnasium torch
    ```
    Alternatively, you can create a `requirements.txt` file with the following content:

    ```
    numpy
    matplotlib
    gymnasium
    torch
    ```

    And then install them using:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the experiments and visualize the results, open and execute the Jupyter Notebook:

```bash
# If you don't have Jupyter installed
pip install jupyter

jupyter notebook
```

Then, navigate to and open `main_notebook.ipynb` (please replace with your actual notebook filename). Run all cells sequentially to reproduce the results and plots.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
