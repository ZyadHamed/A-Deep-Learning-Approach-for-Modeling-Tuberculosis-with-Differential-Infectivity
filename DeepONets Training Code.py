import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from typing import Tuple, Dict, List
import warnings
import os
warnings.filterwarnings('ignore')

# ==================== Model Parameters ====================

def get_params(lambda_val, mu_val, N):
    """Get parameters for specific lambda and mu values"""
    return [
        lambda_val/N,    # lambda (recruitment rate) - normalized by population
        0.025,           # beta (transmission rate)
        1,               # delta (differential infectivity)
        0.3,             # p (fraction that goes directly to infectious)
        mu_val,          # mu (natural death rate) - variable
        0.005,           # k (progression rate from exposed to infectious)
        0,               # r1 (early treatment effectiveness, not used here)
        0.8182,          # r2 (treatment rate of infectious)
        0.02,            # phi (rate from I to L)
        0.01,            # gamma (reactivation from L to I)
        0.0227,          # d1 (death rate from I)
        0.20             # d2 (death rate from L)
    ]

def tb_ode_system(y: np.ndarray, params: List[float], N: float) -> np.ndarray:
    """
    TB epidemiological model ODE system with variable parameters
    y = [S, E, I, L] - population fractions
    """
    S, E, I, L = y
    λ, β, δ, p, μ, k, r1, r2, φ, γ, d1, d2 = params
    
    # ODE system with population scaling
    dSdt = λ - β * S * (I + δ * L) * N - μ * S
    dEdt = β * (1 - p) * S * (I + δ * L) * N + r2 * I - (μ + k * (1 - r1)) * E
    dIdt = β * p * S * (I + δ * L) * N + k * (1 - r1) * E + γ * L - (μ + d1 + φ * (1 - r2) + r2) * I
    dLdt = φ * (1 - r2) * I - (μ + d2 + γ) * L
    
    return np.array([dSdt, dEdt, dIdt, dLdt])

def runge_kutta_4(f, y0: np.ndarray, t: np.ndarray, params: List[float], N: float) -> np.ndarray:
    """4th order Runge-Kutta solver with parameter and population passing"""
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    
    for i in range(1, n):
        h = t[i] - t[i - 1]
        k1 = f(y[i - 1], params, N)
        k2 = f(y[i - 1] + h/2 * k1, params, N)
        k3 = f(y[i - 1] + h/2 * k2, params, N)
        k4 = f(y[i - 1] + h * k3, params, N)
        y[i] = y[i - 1] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Constraint enforcement
        y[i] = np.maximum(y[i], 1e-10)  # Ensure positive values
        y[i] = y[i] / y[i].sum()        # Ensure sum equals 1
    
    return y

def generate_reference_solution(initial_conditions: np.ndarray, time_points: np.ndarray, 
                              params: List[float], N: float) -> np.ndarray:
    """Generate reference solution using RK4 with specific parameters"""
    return runge_kutta_4(tb_ode_system, initial_conditions, time_points, params, N)

# ==================== DeepONet Architecture ====================

class BranchNetwork(nn.Module):
    """Branch network processes initial conditions + parameters"""
    
    def __init__(self, input_dim: int = 6, hidden_dim: int = 256, num_layers: int = 4):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization for better convergence"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class TrunkNetwork(nn.Module):
    """Trunk network processes query points (locations)"""
    
    def __init__(self, input_dim: int = 1, hidden_dim: int = 256, num_layers: int = 4):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization for better convergence"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class DeepONet(nn.Module):
    """
    Deep Operator Network for TB epidemiological model with variable parameters
    Maps [initial_conditions + parameters] -> solution trajectories
    """
    
    def __init__(self, 
                 branch_input: int = 6,       # [S, E, I, L, λ, μ] 
                 trunk_input: int = 1,        # time points
                 hidden_dim: int = 256,
                 num_outputs: int = 4):       # [S, E, I, L] solutions
        super().__init__()
        
        self.branch_net = BranchNetwork(branch_input, hidden_dim)
        self.trunk_net = TrunkNetwork(trunk_input, hidden_dim)
        self.hidden_dim = hidden_dim
        self.num_outputs = num_outputs
        
        # Separate output layers for each compartment
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(num_outputs)
        ])
    
    def forward(self, branch_input: torch.Tensor, time_points: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of DeepONet
        
        Args:
            branch_input: [batch_size, 6] - [S, E, I, L, λ, μ]
            time_points: [batch_size, time_steps, 1] - time points
        
        Returns:
            solutions: [batch_size, time_steps, 4] - predicted [S, E, I, L] trajectories
        """
        batch_size, time_steps, _ = time_points.shape
        
        # Branch network: process initial conditions + parameters
        branch_output = self.branch_net(branch_input)  # [batch_size, hidden_dim]
        
        # Trunk network: process time points
        trunk_input = time_points.reshape(-1, 1)  # [batch_size * time_steps, 1]
        trunk_output = self.trunk_net(trunk_input)  # [batch_size * time_steps, hidden_dim]
        trunk_output = trunk_output.view(batch_size, time_steps, self.hidden_dim)
        
        # Combine branch and trunk outputs
        branch_expanded = branch_output.unsqueeze(1).expand(-1, time_steps, -1)
        combined = branch_expanded * trunk_output  # Element-wise product
        
        # Generate outputs for each compartment
        solutions = torch.zeros(batch_size, time_steps, self.num_outputs, 
                              device=branch_input.device)
        
        for i in range(self.num_outputs):
            output = self.output_layers[i](combined)
            solutions[:, :, i] = output.squeeze(-1)
        
        # Apply softmax to ensure population fractions sum to 1 and are positive
        solutions = F.softmax(solutions, dim=-1)
        
        return solutions

# ==================== Training Framework ====================

class DeepONetTrainer:
    """Training framework for DeepONet TB model with custom initial conditions"""
    
    def __init__(self, time_domain: np.ndarray, epochs: int = 3000, save_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU - training will be slower")
        
        self.model = DeepONet().to(self.device)
        self.time_domain = time_domain
        self.epochs = epochs
        
        # Set save path
        self.save_path = save_path if save_path else r"C:\Users\Mohamed\OneDrive\Desktop\ward"
        
        # Ensure save directory exists
        if not os.path.exists(self.save_path):
            try:
                os.makedirs(self.save_path)
                print(f"Created directory: {self.save_path}")
            except Exception as e:
                print(f"Could not create directory {self.save_path}: {e}")
                self.save_path = "."  # Fallback to current directory
                print(f"Using current directory instead: {os.path.abspath(self.save_path)}")
        else:
            print(f"Save directory: {self.save_path}")
        
        # Training configuration
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=3e-4, 
            weight_decay=5e-5
        )
        
        # FIXED: Removed verbose parameter for compatibility
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.8, 
            patience=300  # REDUCED patience
        )
        
        # Loss configuration
        self.criterion = nn.MSELoss()
        self.lambda_data = 100.0
        self.lambda_constraint = 1000.0
        
        # Training tracking
        self.loss_history = []
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.last_save_epoch = -1  # Track when we last saved
        self.best_loss_improvement_threshold = 0.001  # Only save if improvement is significant
        
        # Parameter ranges for custom initial conditions
        self.lambda_range = (1, 10)      # λ ranges from 1 to 10
        self.mu_range = (0.0101, 0.0227) # μ ranges from 0.0101 to 0.0227
        
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def save_model(self, filename: str = "deeponet_tb_model.pth"):
        """Save the trained model and training information"""
        try:
            full_path = os.path.join(self.save_path, filename)
            
            # Prepare save dictionary
            save_dict = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'loss_history': self.loss_history,
                'best_loss': self.best_loss,
                'time_domain': self.time_domain,
                'lambda_range': self.lambda_range,
                'mu_range': self.mu_range,
                'epochs_completed': len(self.loss_history),
                'model_config': {
                    'branch_input': 6,
                    'trunk_input': 1,
                    'hidden_dim': 256,
                    'num_outputs': 4
                }
            }
            
            torch.save(save_dict, full_path)
            print(f"Model saved successfully to: {full_path}")
            return full_path
            
        except Exception as e:
            print(f"Error saving model: {e}")
            # Try saving to current directory as fallback
            try:
                fallback_path = f"./{filename}"
                torch.save(save_dict, fallback_path)
                print(f"Model saved to fallback location: {os.path.abspath(fallback_path)}")
                return os.path.abspath(fallback_path)
            except Exception as e2:
                print(f"Failed to save model even to fallback location: {e2}")
                return None
    
    def load_model(self, filename: str = "deeponet_tb_model.pth"):
        """Load a previously saved model"""
        try:
            full_path = os.path.join(self.save_path, filename)
            
            if not os.path.exists(full_path):
                print(f"Model file not found: {full_path}")
                return False
            
            checkpoint = torch.load(full_path, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training history
            self.loss_history = checkpoint['loss_history']
            self.best_loss = checkpoint['best_loss']
            self.time_domain = checkpoint['time_domain']
            self.lambda_range = checkpoint['lambda_range']
            self.mu_range = checkpoint['mu_range']
            
            print(f"Model loaded successfully from: {full_path}")
            print(f"Epochs completed: {checkpoint['epochs_completed']}")
            print(f"Best loss: {self.best_loss:.6f}")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def population_constraint_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """Ensure population fractions sum to 1"""
        pop_sums = torch.sum(predictions, dim=-1)  # Sum across compartments
        constraint_loss = torch.mean((pop_sums - 1.0) ** 2)
        return constraint_loss
    
    def positivity_constraint_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """Ensure all population fractions are non-negative"""
        negative_penalty = torch.mean(torch.relu(-predictions) ** 2)
        return negative_penalty
    
    def generate_training_data(self, num_samples: int = 1500):  # REDUCED samples
        """Generate training dataset with custom initial conditions"""
        print(f"Generating {num_samples} samples with custom initial conditions...")
        
        # Storage for training data
        branch_inputs = []  # [S, E, I, L, λ, μ]
        solutions = []
        population_sizes = []  # Store N for each sample
        
        valid_samples = 0
        attempts = 0
        max_attempts = num_samples * 3
        
        while valid_samples < num_samples and attempts < max_attempts:
            attempts += 1
            
            # Sample parameters
            lambda_val = np.random.uniform(*self.lambda_range)
            mu_val = np.random.uniform(*self.mu_range)
            
            # Calculate initial conditions: I=0, L=0, E=1, S=λ/μ
            I_init = 0.0
            L_init = 0.0
            E_init = 1.0
            S_init = lambda_val / mu_val
            
            # Calculate total population N
            N = S_init + E_init + I_init + L_init  # N = λ/μ + 1
            
            # Normalize to get proportions (sum = 1)
            initial_proportions = np.array([S_init, E_init, I_init, L_init]) / N
            
            # Get parameters for this sample
            params = get_params(lambda_val, mu_val, N)
            
            try:
                # Generate reference solution
                solution = generate_reference_solution(initial_proportions, self.time_domain, params, N)
                
                # Validation checks
                if (np.all(solution >= -1e-8) and
                    np.all(solution <= 1.1) and
                    not np.any(np.isnan(solution)) and
                    not np.any(np.isinf(solution)) and
                    np.all(np.abs(solution.sum(axis=1) - 1.0) < 1e-6)):
                    
                    # Normalize λ and μ for branch input
                    lambda_norm = (lambda_val - self.lambda_range[0]) / (self.lambda_range[1] - self.lambda_range[0])
                    mu_norm = (mu_val - self.mu_range[0]) / (self.mu_range[1] - self.mu_range[0])
                    
                    # Create branch input: [S, E, I, L, λ_norm, μ_norm]
                    branch_input = np.concatenate([initial_proportions, [lambda_norm, mu_norm]])
                    
                    branch_inputs.append(branch_input)
                    solutions.append(solution)
                    population_sizes.append(N)
                    valid_samples += 1
                    
            except Exception as e:
                continue
        
        print(f"Generated {valid_samples} valid samples out of {attempts} attempts")
        
        # Convert to tensors
        self.train_branch = torch.tensor(
            branch_inputs, 
            dtype=torch.float32, 
            device=self.device
        )
        self.train_sol = torch.tensor(
            solutions, 
            dtype=torch.float32, 
            device=self.device
        )
        self.train_time = torch.tensor(
            self.time_domain.reshape(1, -1, 1), 
            dtype=torch.float32, 
            device=self.device
        ).repeat(self.train_branch.shape[0], 1, 1)
        
        # Store population sizes for later use (keep order matching training data)
        self.population_sizes = population_sizes
        
        print(f"Training data generated:")
        print(f"Branch inputs: {self.train_branch.shape}")
        print(f"Solutions: {self.train_sol.shape}")
        print(f"Time points: {self.train_time.shape}")
        print(f"Population sizes range: {min(population_sizes):.2f} - {max(population_sizes):.2f}")
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute weighted loss with constraints - using population numbers for MSE"""
        batch_size = predictions.shape[0]
        
        # Convert proportions to population numbers for each sample
        # Get population sizes for each sample in the batch
        population_sizes = torch.tensor(
            [self.population_sizes[i] for i in range(batch_size)], 
            dtype=torch.float32, 
            device=predictions.device
        ).unsqueeze(1).unsqueeze(2)  # Shape: [batch_size, 1, 1]
        
        # Convert to population numbers
        pred_numbers = predictions * population_sizes  # Broadcasting
        target_numbers = targets * population_sizes
        
        # Individual compartment losses (on population numbers)
        loss_S = self.criterion(pred_numbers[:, :, 0], target_numbers[:, :, 0])
        loss_E = self.criterion(pred_numbers[:, :, 1], target_numbers[:, :, 1])
        loss_I = self.criterion(pred_numbers[:, :, 2], target_numbers[:, :, 2])
        loss_L = self.criterion(pred_numbers[:, :, 3], target_numbers[:, :, 3])
        
        # Weighted data loss
        data_loss = loss_S + loss_E + loss_I + loss_L * 10
        
        # Constraint losses (still on proportions to ensure they sum to 1)
        constraint_loss = self.population_constraint_loss(predictions)
        positivity_loss = self.positivity_constraint_loss(predictions)
        
        # Total loss
        total_loss = (
            self.lambda_data * data_loss + 
            self.lambda_constraint * (constraint_loss + positivity_loss)
        )
        
        return {
            'total': total_loss,
            'data': data_loss,
            'constraint': constraint_loss,
            'positivity': positivity_loss,
            'S': loss_S,
            'E': loss_E,
            'I': loss_I,
            'L': loss_L
        }
    
    def validate_predictions(self, predictions: torch.Tensor) -> Dict[str, float]:
        """Validate prediction constraints"""
        if len(predictions.shape) == 2:
            predictions = predictions.unsqueeze(0)
        
        # Population sum validation
        sums = torch.sum(predictions, dim=-1)
        
        validation_metrics = {
            'mean_sum': torch.mean(sums).item(),
            'min_sum': torch.min(sums).item(),
            'max_sum': torch.max(sums).item(),
            'sum_deviation': torch.mean(torch.abs(sums - 1)).item(),
            'negative_count': torch.sum(predictions < 0).item(),
        }
        
        # Overall validity check
        validation_metrics['valid'] = (
            torch.all(sums > 0.99) and 
            torch.all(sums < 1.01) and 
            validation_metrics['negative_count'] == 0
        )
        
        return validation_metrics
    
    def train(self, num_samples: int = 1500):  # REDUCED samples
        """Main training loop"""
        self.generate_training_data(num_samples)
        
        print(f"Starting training for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            self.model.train()
            
            # Forward pass
            predictions = self.model(self.train_branch, self.train_time)
            
            # Compute losses
            losses = self.compute_loss(predictions, self.train_sol)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            
            # Update scheduler
            self.scheduler.step(losses['total'])
            self.loss_history.append(losses['total'].item())
            
            # Early stopping check with smarter saving
            current_loss = losses['total'].item()
            if current_loss < self.best_loss:
                loss_improvement = self.best_loss - current_loss
                self.best_loss = current_loss
                self.patience_counter = 0
                
                # Save model only under specific conditions:
                should_save = (
                    epoch > 1000 and  # After significant training
                    (epoch - self.last_save_epoch) >= 500 and  # At least 500 epochs since last save
                    loss_improvement > self.best_loss_improvement_threshold and  # Significant improvement
                    epoch % 500 == 0  # Only on specific intervals
                )
                
                if should_save:
                    print(f"Saving best model (improvement: {loss_improvement:.6f})")
                    self.save_model("best_deeponet_tb_model.pth")
                    self.last_save_epoch = epoch
            else:
                self.patience_counter += 1
            
            if self.patience_counter > 1000:  # REDUCED patience
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            # Periodic logging and checkpoint saving
            if (epoch + 1) % 500 == 0:  # More frequent logging
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:5d} | "
                      f"Loss: {losses['total'].item():.6f} | "
                      f"Data: {losses['data'].item():.6f} | "
                      f"Constraint: {losses['constraint'].item():.6f} | "
                      f"LR: {current_lr:.6f}")
                
                # Validate sample prediction
                with torch.no_grad():
                    sample_pred = predictions[:1]
                    validation = self.validate_predictions(sample_pred)
                    print(f"Validation - Sum: {validation['mean_sum']:.6f}, "
                          f"Deviation: {validation['sum_deviation']:.6f}, "
                          f"Valid: {validation['valid']}")
                
                # Save checkpoint at major milestones
                if (epoch + 1) in [1000, 2000, 4000, 6000]:  # Specific milestone epochs
                    checkpoint_name = f"checkpoint_epoch_{epoch+1}.pth"
                    print(f"Saving checkpoint at epoch {epoch+1}")
                    self.save_model(checkpoint_name)
        
        print(f"Training completed after {len(self.loss_history)} epochs")
        print(f"Best loss achieved: {self.best_loss:.6f}")
        
        # Save final model
        final_path = self.save_model("final_deeponet_tb_model.pth")
        return final_path
    
    def predict(self, lambda_val: float, mu_val: float, time_points: np.ndarray) -> Tuple[np.ndarray, float]:
        """Make predictions with the trained model for given λ and μ"""
        self.model.eval()
        
        with torch.no_grad():
            # Calculate initial conditions: I=0, L=0, E=1, S=λ/μ
            I_init = 0.0
            L_init = 0.0
            E_init = 1.0
            S_init = lambda_val / mu_val
            
            # Calculate total population N
            N = S_init + E_init + I_init + L_init  # N = λ/μ + 1
            
            # Normalize to get proportions
            initial_proportions = np.array([S_init, E_init, I_init, L_init]) / N
            
            # Normalize parameters for branch input
            lambda_norm = (lambda_val - self.lambda_range[0]) / (self.lambda_range[1] - self.lambda_range[0])
            mu_norm = (mu_val - self.mu_range[0]) / (self.mu_range[1] - self.mu_range[0])
            
            # Create branch input
            branch_input = np.concatenate([initial_proportions, [lambda_norm, mu_norm]])
            branch_tensor = torch.tensor(
                branch_input, 
                dtype=torch.float32, 
                device=self.device
            ).unsqueeze(0)
            
            time_tensor = torch.tensor(
                time_points.reshape(1, -1, 1), 
                dtype=torch.float32, 
                device=self.device
            )
            
            predictions = self.model(branch_tensor, time_tensor)
            
            return predictions.cpu().numpy().squeeze(0), N

# ==================== Visualization Functions ====================

def plot_results(lambda_val: float, mu_val: float, 
                time_domain: np.ndarray, 
                predictions: np.ndarray, 
                N: float,
                title: str = "DeepONet TB Model Results"):
    """Plot comparison between predicted and reference solutions in population numbers"""
    
    # Calculate initial conditions and parameters for reference
    I_init = 0.0
    L_init = 0.0
    E_init = 1.0
    S_init = lambda_val / mu_val
    initial_proportions = np.array([S_init, E_init, I_init, L_init]) / N
    params = get_params(lambda_val, mu_val, N)
    
    # Generate reference solution
    reference = generate_reference_solution(initial_proportions, time_domain, params, N)
    
    # Convert percentages to actual population numbers
    pred_numbers = predictions * N
    ref_numbers = reference * N
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    compartments = ['S', 'E', 'I', 'L']
    colors = ['blue', 'orange', 'red', 'green']
    full_names = ['Susceptible', 'Exposed', 'Infectious', 'Latent']
    
    for i, (ax, comp, color, full_name) in enumerate(zip(axes.flat, compartments, colors, full_names)):
        # Plot predictions and reference
        ax.plot(time_domain, pred_numbers[:, i], 
               label=f'{comp} Predicted', color=color, linewidth=2)
        ax.plot(time_domain, ref_numbers[:, i], 
               label=f'{comp} Reference', linestyle='--', color='black', alpha=0.7)
        
        ax.set_title(f'{comp}(t) - {full_name}')
        ax.set_xlabel('Time (years)')
        ax.set_ylabel(f'{comp} (Number of People)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.suptitle(f'{title}\nλ={lambda_val:.2f}, μ={mu_val:.4f}, N={round(N)}', fontsize=14, y=1.02)
    plt.show()

def plot_mse_analysis(lambda_val: float, mu_val: float,
                     time_domain: np.ndarray,
                     predictions: np.ndarray,
                     N: float,
                     title: str = "MSE Error Analysis"):
    """Plot MSE analysis in population numbers"""
    
    # Calculate initial conditions and parameters for reference
    I_init = 0.0
    L_init = 0.0
    E_init = 1.0
    S_init = lambda_val / mu_val
    initial_proportions = np.array([S_init, E_init, I_init, L_init]) / N
    params = get_params(lambda_val, mu_val, N)
    
    # Generate reference solution
    reference = generate_reference_solution(initial_proportions, time_domain, params, N)
    
    # Convert to population numbers for MSE calculation
    pred_numbers = predictions * N
    ref_numbers = reference * N
    
    # Calculate MSE (Mean Squared Error) in population numbers
    squared_errors = (pred_numbers - ref_numbers) ** 2
    
    plt.figure(figsize=(12, 8))
    compartments = ['S', 'E', 'I', 'L']
    colors = ['blue', 'orange', 'red', 'green']
    
    for i, (comp, color) in enumerate(zip(compartments, colors)):
        plt.plot(time_domain, squared_errors[:, i], 
                'o-', label=f'{comp}', color=color, markersize=3, linewidth=2)
    
    plt.xlabel('Time (years)', fontsize=12)
    plt.ylabel('Mean Squared Error (Population Numbers)', fontsize=12)
    plt.title(f'{title}\nλ={lambda_val:.2f}, μ={mu_val:.4f}, N={round(N)}', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print MSE summary
    mean_mse = np.mean(squared_errors, axis=0)
    print(f"\nMSE Summary (Population Numbers):")
    print("-" * 40)
    for i, comp in enumerate(compartments):
        print(f"{comp}: Mean MSE = {mean_mse[i]:.6f}")

# ==================== Main Execution ====================

def main():
    """Main execution function"""
    print("=" * 60)
    print("DeepONet for TB Model")
    print("Compatible with older PyTorch versions")
    print("Auto-saves to specified directory")
    print("=" * 60)
    
    # REDUCED Configuration for faster testing
    time_domain = np.linspace(0, 25, 100)  # 25 years, 100 points
    epochs = 8000  # REDUCED
    num_samples = 2000  # REDUCED
    
    # Custom save path
    save_path = r"C:\Users\Mohamed\OneDrive\Desktop\ward"
    
    print(f"Configuration:")
    print(f"   Time domain: 0-25 years, {len(time_domain)} points")
    print(f"   Training epochs: {epochs}")
    print(f"   Training samples: {num_samples}")
    print(f"   Save path: {save_path}")
    
    # Initialize and train model
    trainer = DeepONetTrainer(time_domain, epochs, save_path)
    
    try:
        final_model_path = trainer.train(num_samples)
        
        # Test with specific parameters (only one test case)
        test_lambda = 5.0    # Example λ value
        test_mu = 0.015      # Example μ value
        
        print(f"\nTesting with parameters:")
        print(f"   λ = {test_lambda}")
        print(f"   μ = {test_mu}")
        
        # Calculate expected initial conditions
        S_init = test_lambda / test_mu
        N = S_init + 1  # N = λ/μ + 1 (since E=1, I=0, L=0)
        print(f"   Expected N = {N:.2f}")
        print(f"   Initial conditions: S={S_init:.2f}, E=1, I=0, L=0")
        print(f"   Initial proportions: S={S_init/N:.4f}, E={1/N:.4f}, I=0, L=0")
        
        # Make predictions
        predictions, N_actual = trainer.predict(test_lambda, test_mu, time_domain)
        
        # Validate predictions
        validation = trainer.validate_predictions(torch.tensor(predictions))
        print(f"\nPrediction Validation:")
        print("-" * 30)
        for key, value in validation.items():
            status = "PASSED" if key == 'valid' and value else "FAILED" if key == 'valid' else ""
            print(f"{key:15s}: {value} {status}")
        
        # Create visualizations
        plot_results(test_lambda, test_mu, time_domain, predictions, N_actual,
                    "DeepONet TB Model - Population Dynamics")
        
        plot_mse_analysis(test_lambda, test_mu, time_domain, predictions, N_actual,
                         "DeepONet TB Model - MSE Error Analysis")
        
        print(f"\nTraining Summary:")
        print("-" * 30)
        print(f"Training samples: {num_samples}")
        print(f"Final validation: {'PASSED' if validation['valid'] else 'FAILED'}")
        print(f"Training epochs: {len(trainer.loss_history)}")
        print(f"Best loss: {trainer.best_loss:.6f}")
        print(f"Population size N: {round(N_actual)}")
        if final_model_path:
            print(f"Model saved to: {final_model_path}")
        
        # Save a final summary report
        try:
            summary_path = os.path.join(save_path, "training_summary.txt")
            with open(summary_path, 'w') as f:
                f.write("DeepONet TB Model Training Summary\n")
                f.write("=" * 40 + "\n")
                f.write(f"Training completed: {len(trainer.loss_history)} epochs\n")
                f.write(f"Best loss: {trainer.best_loss:.6f}\n")
                f.write(f"Training samples: {num_samples}\n")
                f.write(f"Time domain: 0-25 years, {len(time_domain)} points\n")
                f.write(f"Final validation passed: {validation['valid']}\n")
                f.write("\nTest Results:\n")
                f.write(f"Test (λ={test_lambda}, μ={test_mu}): N={N_actual:.2f}\n")
            print(f"Training summary saved to: {summary_path}")
        except Exception as e:
            print(f"Could not save summary: {e}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        print(f"Suggestions:")
        print(f"   • Try reducing num_samples further (e.g., 500-1000)")
        print(f"   • Reduce epochs (e.g., 1000-2000)")
        print(f"   • Check PyTorch installation")
        print(f"   • Ensure sufficient memory available")
        print(f"   • Check if save directory is accessible")

# Quick test function for initial validation
def quick_test():
    """Quick test with minimal configuration"""
    print("Running quick test...")
    
    time_domain = np.linspace(0, 10, 20)  # Minimal time domain
    epochs = 500  # Very few epochs
    num_samples = 200  # Very few samples
    save_path = r"C:\Users\Mohamed\OneDrive\Desktop\ward"
    
    try:
        trainer = DeepONetTrainer(time_domain, epochs, save_path)
        trainer.train(num_samples)
        
        # Test prediction
        predictions, N = trainer.predict(5.0, 0.015, time_domain)
        validation = trainer.validate_predictions(torch.tensor(predictions))
        
        # Save the quick test model
        model_path = trainer.save_model("quick_test_model.pth")

        print(f"Quick test completed!")
        print(f"   Validation passed: {validation['valid']}")
        print(f"   Population sum: {validation['mean_sum']:.4f}")
        if model_path:
            print(f"   Model saved to: {model_path}")
        
        return validation['valid']
        
    except Exception as e:
        print(f"Quick test failed: {e}")
        return False

def load_and_test(model_filename: str = "deeponet_tb_model.pth"):
    """Load a saved model and run tests"""
    save_path = r"C:\Users\Mohamed\OneDrive\Desktop\ward"
    time_domain = np.linspace(0, 25, 100)
    
    print(f"Loading model: {model_filename}")
    
    # Create trainer instance
    trainer = DeepONetTrainer(time_domain, save_path=save_path)
    
    # Load the model
    if trainer.load_model(model_filename):
        print("Model loaded successfully!")
        
        # Test with parameters
        test_lambda = 5.0
        test_mu = 0.015
        
        predictions, N = trainer.predict(test_lambda, test_mu, time_domain)
        validation = trainer.validate_predictions(torch.tensor(predictions))
        
        print(f"Test results:")
        print(f"   Validation passed: {validation['valid']}")
        print(f"   Population sum: {validation['mean_sum']:.4f}")
        print(f"   Population N: {round(N)}")
        
        # Create visualization
        plot_results(test_lambda, test_mu, time_domain, predictions, N,
                    "Loaded DeepONet TB Model Test")
        
        return True
    else:
        print("Failed to load model")
        return False

if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            quick_test()
        elif sys.argv[1] == "--load":
            model_name = sys.argv[2] if len(sys.argv) > 2 else "final_deeponet_tb_model.pth"
            load_and_test(model_name)
        else:
            print("Usage:")
            print("  python script.py          # Run full training")
            print("  python script.py --quick  # Run quick test")
            print("  python script.py --load [model_name]  # Load and test model")
    else:
        main()