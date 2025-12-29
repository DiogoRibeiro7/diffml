"""Advanced Differential Machine Learning techniques.

This module implements cutting-edge techniques including deep hedging,
reinforcement learning-based strategies, adversarial training, and
meta-learning approaches for option pricing and risk management.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .networks import FeedForwardNet, DifferentialNet
from .config import get_device
from .losses import dml_loss


class DeepHedgingNet(nn.Module):
    """Deep Hedging neural network for optimal hedging strategies.

    Deep hedging learns optimal hedging strategies directly from data
    without relying on model assumptions like Black-Scholes.

    References:
        Buehler et al. (2019): "Deep Hedging"
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        use_lstm: bool = True
    ):
        """Initialize Deep Hedging network.

        Parameters:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (hedge ratios)
            use_lstm: Whether to use LSTM for path dependency
        """
        super().__init__()

        self.use_lstm = use_lstm

        if use_lstm:
            # LSTM for capturing path dependency
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dims[0],
                num_layers=2,
                batch_first=True,
                dropout=0.1
            )
            input_dim_fc = hidden_dims[0]
        else:
            input_dim_fc = input_dim

        # Feedforward layers for hedge ratio
        layers = []
        prev_dim = input_dim_fc

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.hedge_network = nn.Sequential(*layers)

        # Value function network (critic)
        self.value_network = FeedForwardNet(
            input_dim=input_dim_fc,
            hidden_dims=hidden_dims,
            output_dim=1
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """Forward pass for deep hedging.

        Parameters:
            x: Input features (batch_size, seq_len, input_dim) if LSTM
               or (batch_size, input_dim) if feedforward
            hidden: Hidden state for LSTM

        Returns:
            Tuple of (hedge_ratios, value_estimate, hidden_state)
        """
        if self.use_lstm and x.dim() == 3:
            # Process with LSTM
            lstm_out, hidden = self.lstm(x, hidden)
            # Use last output for hedge decision
            features = lstm_out[:, -1, :]
        else:
            features = x
            hidden = None

        # Compute hedge ratios
        hedge_ratios = self.hedge_network(features)

        # Compute value estimate
        value = self.value_network(features)

        return hedge_ratios, value, hidden


class RLHedgingAgent:
    """Reinforcement Learning agent for dynamic hedging.

    Uses Deep Q-Learning or Policy Gradient methods to learn
    optimal hedging policies in a model-free manner.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        device: Optional[torch.device] = None
    ):
        """Initialize RL hedging agent.

        Parameters:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate
            gamma: Discount factor
            device: Torch device
        """
        self.device = device or get_device()
        self.gamma = gamma

        # Actor network (policy)
        self.actor = FeedForwardNet(
            input_dim=state_dim,
            hidden_dims=[128, 64, 32],
            output_dim=action_dim
        ).to(self.device)

        # Critic network (value function)
        self.critic = FeedForwardNet(
            input_dim=state_dim,
            hidden_dims=[128, 64, 32],
            output_dim=1
        ).to(self.device)

        # Target networks for stability
        self.target_actor = FeedForwardNet(
            input_dim=state_dim,
            hidden_dims=[128, 64, 32],
            output_dim=action_dim
        ).to(self.device)

        self.target_critic = FeedForwardNet(
            input_dim=state_dim,
            hidden_dims=[128, 64, 32],
            output_dim=1
        ).to(self.device)

        # Copy weights to target
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=learning_rate
        )

        # Replay buffer
        self.replay_buffer = []
        self.buffer_size = 100000

    def select_action(self, state: torch.Tensor, epsilon: float = 0.0) -> torch.Tensor:
        """Select hedging action using epsilon-greedy policy.

        Parameters:
            state: Current state
            epsilon: Exploration rate

        Returns:
            Selected action (hedge ratio)
        """
        if np.random.random() < epsilon:
            # Explore: random action
            action = torch.randn(self.actor.output_dim, device=self.device)
        else:
            # Exploit: use policy
            with torch.no_grad():
                action = self.actor(state)

        return action

    def store_transition(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool
    ):
        """Store transition in replay buffer.

        Parameters:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode done flag
        """
        if len(self.replay_buffer) >= self.buffer_size:
            self.replay_buffer.pop(0)

        self.replay_buffer.append((
            state.cpu(), action.cpu(), reward,
            next_state.cpu(), done
        ))

    def train(self, batch_size: int = 64) -> Dict[str, float]:
        """Train the RL agent using experience replay.

        Parameters:
            batch_size: Batch size for training

        Returns:
            Dictionary with loss values
        """
        if len(self.replay_buffer) < batch_size:
            return {}

        # Sample batch
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]

        states = torch.stack([b[0] for b in batch]).to(self.device)
        actions = torch.stack([b[1] for b in batch]).to(self.device)
        rewards = torch.tensor([b[2] for b in batch], device=self.device)
        next_states = torch.stack([b[3] for b in batch]).to(self.device)
        dones = torch.tensor([b[4] for b in batch], device=self.device)

        # Critic loss (TD error)
        current_q = self.critic(states).squeeze()
        with torch.no_grad():
            next_q = self.target_critic(next_states).squeeze()
            target_q = rewards + self.gamma * next_q * (~dones)

        critic_loss = F.mse_loss(current_q, target_q)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss (policy gradient)
        actor_loss = -self.critic(states).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        tau = 0.001
        for target_param, param in zip(
            self.target_actor.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )

        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }


class AdversarialDML(nn.Module):
    """Adversarial training for robust DML.

    Uses adversarial examples to improve robustness of pricing
    and hedging models against market perturbations.
    """

    def __init__(
        self,
        base_model: nn.Module,
        epsilon: float = 0.01,
        alpha: float = 0.001,
        num_steps: int = 10
    ):
        """Initialize adversarial DML.

        Parameters:
            base_model: Base pricing model
            epsilon: Maximum perturbation size
            alpha: Step size for PGD attack
            num_steps: Number of PGD steps
        """
        super().__init__()
        self.base_model = base_model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps

    def pgd_attack(
        self,
        x: torch.Tensor,
        y_price: torch.Tensor,
        y_delta: torch.Tensor,
        lambda_val: float = 1.0
    ) -> torch.Tensor:
        """Generate adversarial examples using PGD.

        Parameters:
            x: Input features
            y_price: True prices
            y_delta: True deltas
            lambda_val: Weight for delta term

        Returns:
            Adversarial examples
        """
        x_adv = x.clone().detach().requires_grad_(True)

        for _ in range(self.num_steps):
            # Forward pass
            pred_price = self.base_model(x_adv)

            # Compute gradients for delta
            pred_delta = torch.autograd.grad(
                pred_price.sum(), x_adv,
                create_graph=True, retain_graph=True
            )[0][:, 0:1]  # Take first component as delta

            # Compute loss
            loss = dml_loss(
                pred_price, y_price, pred_delta, y_delta,
                lambda_val=lambda_val
            )

            # Compute gradients
            loss.backward()

            # Update adversarial example
            with torch.no_grad():
                x_adv = x_adv + self.alpha * x_adv.grad.sign()
                # Project back to epsilon ball
                x_adv = torch.max(torch.min(x_adv, x + self.epsilon), x - self.epsilon)
                x_adv = x_adv.detach().requires_grad_(True)

        return x_adv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through base model.

        Parameters:
            x: Input features

        Returns:
            Model predictions
        """
        return self.base_model(x)

    def robust_train_step(
        self,
        x: torch.Tensor,
        y_price: torch.Tensor,
        y_delta: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        lambda_val: float = 1.0,
        adv_weight: float = 0.5
    ) -> Dict[str, float]:
        """Perform robust training step with adversarial examples.

        Parameters:
            x: Input features
            y_price: True prices
            y_delta: True deltas
            optimizer: Optimizer
            lambda_val: Weight for delta term
            adv_weight: Weight for adversarial loss

        Returns:
            Dictionary with loss values
        """
        # Generate adversarial examples
        x_adv = self.pgd_attack(x, y_price, y_delta, lambda_val)

        # Clean predictions
        pred_price_clean = self.base_model(x)
        pred_delta_clean = torch.autograd.grad(
            pred_price_clean.sum(), x,
            create_graph=True, retain_graph=True
        )[0][:, 0:1]

        # Adversarial predictions
        pred_price_adv = self.base_model(x_adv)
        pred_delta_adv = torch.autograd.grad(
            pred_price_adv.sum(), x_adv,
            create_graph=True, retain_graph=True
        )[0][:, 0:1]

        # Compute losses
        loss_clean = dml_loss(
            pred_price_clean, y_price, pred_delta_clean, y_delta,
            lambda_val=lambda_val
        )

        loss_adv = dml_loss(
            pred_price_adv, y_price, pred_delta_adv, y_delta,
            lambda_val=lambda_val
        )

        # Combined loss
        total_loss = (1 - adv_weight) * loss_clean + adv_weight * loss_adv

        # Update model
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'clean_loss': loss_clean.item(),
            'adv_loss': loss_adv.item()
        }


class MetaLearningDML(nn.Module):
    """Meta-learning for fast adaptation to new option types.

    Uses Model-Agnostic Meta-Learning (MAML) to quickly adapt
    to new option types with few samples.
    """

    def __init__(
        self,
        base_model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        num_inner_steps: int = 5
    ):
        """Initialize meta-learning DML.

        Parameters:
            base_model: Base model architecture
            inner_lr: Learning rate for inner loop
            outer_lr: Learning rate for outer loop
            num_inner_steps: Number of inner gradient steps
        """
        super().__init__()
        self.base_model = base_model
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps

        # Meta-optimizer
        self.meta_optimizer = torch.optim.Adam(
            self.base_model.parameters(), lr=outer_lr
        )

    def inner_loop(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        model: nn.Module
    ) -> nn.Module:
        """Perform inner loop adaptation.

        Parameters:
            support_x: Support set features
            support_y: Support set targets
            model: Model to adapt

        Returns:
            Adapted model
        """
        # Clone model for task-specific adaptation
        adapted_model = type(model)(
            model.input_dim,
            model.hidden_dims,
            model.output_dim
        ).to(support_x.device)

        adapted_model.load_state_dict(model.state_dict())

        # Task-specific optimizer
        task_optimizer = torch.optim.SGD(
            adapted_model.parameters(), lr=self.inner_lr
        )

        # Adaptation steps
        for _ in range(self.num_inner_steps):
            pred = adapted_model(support_x)
            loss = F.mse_loss(pred, support_y)

            task_optimizer.zero_grad()
            loss.backward()
            task_optimizer.step()

        return adapted_model

    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor
    ) -> torch.Tensor:
        """Meta-learning forward pass.

        Parameters:
            support_x: Support set features
            support_y: Support set targets
            query_x: Query set features

        Returns:
            Predictions on query set
        """
        # Adapt model on support set
        adapted_model = self.inner_loop(support_x, support_y, self.base_model)

        # Evaluate on query set
        with torch.no_grad():
            predictions = adapted_model(query_x)

        return predictions

    def meta_train_step(
        self,
        tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> float:
        """Perform meta-training step.

        Parameters:
            tasks: List of (support_x, support_y, query_x, query_y) tuples

        Returns:
            Meta-loss value
        """
        meta_loss = 0.0

        for support_x, support_y, query_x, query_y in tasks:
            # Adapt model on support set
            adapted_model = self.inner_loop(support_x, support_y, self.base_model)

            # Evaluate on query set
            query_pred = adapted_model(query_x)
            task_loss = F.mse_loss(query_pred, query_y)

            meta_loss += task_loss

        # Meta-update
        meta_loss = meta_loss / len(tasks)

        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()


class NeuralSDE(nn.Module):
    """Neural Stochastic Differential Equations for market dynamics.

    Learns market dynamics directly from data using neural SDEs,
    enabling more flexible and realistic price modeling.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        num_layers: int = 3
    ):
        """Initialize Neural SDE.

        Parameters:
            state_dim: Dimension of state space
            hidden_dim: Hidden layer dimension
            num_layers: Number of layers
        """
        super().__init__()

        # Drift network (mu)
        self.drift_net = FeedForwardNet(
            input_dim=state_dim + 1,  # State + time
            hidden_dims=[hidden_dim] * num_layers,
            output_dim=state_dim
        )

        # Diffusion network (sigma)
        self.diffusion_net = FeedForwardNet(
            input_dim=state_dim + 1,  # State + time
            hidden_dims=[hidden_dim] * num_layers,
            output_dim=state_dim
        )

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute drift and diffusion.

        Parameters:
            t: Time
            x: State

        Returns:
            Tuple of (drift, diffusion)
        """
        # Concatenate time and state
        tx = torch.cat([t.unsqueeze(-1), x], dim=-1)

        drift = self.drift_net(tx)
        diffusion = self.diffusion_net(tx)

        # Ensure positive diffusion
        diffusion = F.softplus(diffusion)

        return drift, diffusion

    def sample_paths(
        self,
        x0: torch.Tensor,
        ts: torch.Tensor,
        num_samples: int
    ) -> torch.Tensor:
        """Sample paths from the Neural SDE.

        Parameters:
            x0: Initial state
            ts: Time points
            num_samples: Number of paths to sample

        Returns:
            Sampled paths
        """
        device = x0.device
        dt = ts[1] - ts[0]  # Assume uniform time grid

        paths = []
        x = x0.unsqueeze(0).repeat(num_samples, 1)

        for t in ts:
            drift, diffusion = self.forward(
                t.expand(num_samples), x
            )

            # Euler-Maruyama step
            dW = torch.randn_like(x) * torch.sqrt(dt)
            x = x + drift * dt + diffusion * dW

            paths.append(x.unsqueeze(1))

        return torch.cat(paths, dim=1)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Advanced DML Techniques")
    print("="*60)

    device = get_device()

    # Test 1: Deep Hedging
    print("\n1. Deep Hedging Network")
    deep_hedge = DeepHedgingNet(
        input_dim=5,
        hidden_dims=[64, 32],
        output_dim=1,
        use_lstm=False
    ).to(device)

    x = torch.randn(32, 5, device=device)
    hedge_ratios, values, _ = deep_hedge(x)
    print(f"   Hedge ratios shape: {hedge_ratios.shape}")
    print(f"   Values shape: {values.shape}")

    # Test 2: RL Hedging Agent
    print("\n2. Reinforcement Learning Agent")
    rl_agent = RLHedgingAgent(
        state_dim=5,
        action_dim=1,
        device=device
    )

    state = torch.randn(1, 5, device=device)
    action = rl_agent.select_action(state)
    print(f"   Selected action: {action.item():.4f}")

    # Test 3: Adversarial DML
    print("\n3. Adversarial Training")
    base_model = FeedForwardNet(5, [32, 16], 1).to(device)
    adv_dml = AdversarialDML(base_model).to(device)

    x = torch.randn(16, 5, device=device, requires_grad=True)
    y_price = torch.randn(16, 1, device=device)
    y_delta = torch.randn(16, 1, device=device)

    x_adv = adv_dml.pgd_attack(x, y_price, y_delta)
    print(f"   Max perturbation: {(x_adv - x).abs().max():.4f}")

    # Test 4: Meta-Learning
    print("\n4. Meta-Learning DML")
    base_model = FeedForwardNet(5, [32, 16], 1).to(device)
    meta_dml = MetaLearningDML(base_model).to(device)

    support_x = torch.randn(10, 5, device=device)
    support_y = torch.randn(10, 1, device=device)
    query_x = torch.randn(5, 5, device=device)

    predictions = meta_dml(support_x, support_y, query_x)
    print(f"   Predictions shape: {predictions.shape}")

    # Test 5: Neural SDE
    print("\n5. Neural SDE")
    neural_sde = NeuralSDE(
        state_dim=2,
        hidden_dim=32,
        num_layers=3
    ).to(device)

    t = torch.tensor(0.0, device=device)
    x = torch.randn(16, 2, device=device)
    drift, diffusion = neural_sde(t, x)
    print(f"   Drift shape: {drift.shape}")
    print(f"   Diffusion shape: {diffusion.shape}")

    print("\n✓ All advanced techniques tested successfully!")