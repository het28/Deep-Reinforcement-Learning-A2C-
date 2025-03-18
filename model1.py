import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# Define SharedAdam optimizer for shared memory updates
class SharedAdam(Adam):
    def __init__(self, *args, **kwargs):
        super(SharedAdam, self).__init__(*args, **kwargs)
        for group in self.param_groups:
            for param in group['params']:
                if param.requires_grad:
                    param.grad = param.data.new(param.size()).zero_()

class ActorCritic(nn.Module):
    def __init__(self, n_states, n_actions):
        super(ActorCritic, self).__init__()

        # Enhanced shared network architecture for larger state space
        self.shared = nn.Sequential(
            nn.Linear(n_states, 256),  # Increased hidden size
            nn.LeakyReLU(),
            nn.Linear(256, 256),      # Added an additional hidden layer
            nn.LeakyReLU(),
            nn.Linear(256, 128),      # Final shared layer
            nn.LeakyReLU()
        )

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(128, n_actions),
            nn.Softmax(dim=-1)
        )

        # Critic network
        self.critic = nn.Linear(128, 1)

        # Initialize weights
        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.actor[0].weight)
        nn.init.zeros_(self.actor[0].bias)

        nn.init.xavier_uniform_(self.critic.weight)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x):
        shared_out = self.shared(x)
        action_probs = self.actor(shared_out)
        state_values = self.critic(shared_out)
        return action_probs, state_values

def compute_loss(action_probs, state_values, actions, rewards, entropy_coeff = 0.05):
    """
    Compute the loss for Actor-Critic training.

    Args:
        action_probs: Probabilities of actions from the actor network.
        state_values: Predicted state values from the critic network.
        actions: Actions taken during training.
        rewards: Rewards received.
        entropy_coeff: Coefficient for entropy regularization.

    Returns:
        Combined loss for policy and value function.
    """
    # Policy loss
    action_dist = torch.distributions.Categorical(probs=action_probs)
    log_probs = action_dist.log_prob(actions)
    entropy = action_dist.entropy().mean()
    advantages = rewards - state_values.squeeze(-1)
    policy_loss = -(log_probs * rewards.detach()).mean() - entropy_coeff * entropy

    # Value loss
    value_loss = F.mse_loss(state_values.squeeze(-1), rewards)

    # Combine losses
    loss = policy_loss + 0.5 * value_loss
    return loss, policy_loss, entropy

def setup_optimizer(model):
    """
    Set up a shared Adam optimizer for the model.

    Args:
        model: The ActorCritic model.

    Returns:
        optimizer: SharedAdam optimizer instance.
    """
    optimizer = SharedAdam(model.parameters(), lr=0.0003, weight_decay=1e-5)
    return optimizer

# Function to encourage exploration dynamically
def exploration_strategy(action_probs, exploration_rate=0.1):
    """
    Modify action probabilities to encourage exploration.

    Args:
        action_probs: Original action probabilities.
        exploration_rate: Probability of choosing a random action.

    Returns:
        Modified action probabilities.
    """
    num_actions = action_probs.size(-1)
    uniform_probs = torch.ones_like(action_probs) / num_actions
    return (1 - exploration_rate) * action_probs + exploration_rate * uniform_probs
