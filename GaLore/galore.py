import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import math
import time

def randomized_svd(A, rank, oversampling=15):
    if rank >= min(A.shape):
        return torch.linalg.svd(A, full_matrices=False)
    k = rank + oversampling
    k = min(k, A.shape[0], A.shape[1])

    P = torch.randn(A.shape[1], k, device=A.device, dtype=A.dtype)
    Z = A @ P
    Q, _ = torch.linalg.qr(Z)
    Y = Q.T @ A
    U_y, S, Vh = torch.linalg.svd(Y, full_matrices=False)
    U = Q @ U_y
    
    return U[:, :rank], S[:rank], Vh[:rank, :]

class Galore(Optimizer): 
    def __init__(self, params, lr=0.001, rank=16, subspace_change_frequency=200, 
                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0, svd_method='exact'):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate {lr}")
        
        defaults = dict(lr=lr, rank=rank, subspace_change_frequency=subspace_change_frequency, 
                        betas=betas, eps=eps, weight_decay=weight_decay, svd_method=svd_method) 
        
        super(Galore, self).__init__(params, defaults)
        self.state['step'] = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None 
        if closure is not None: 
            with torch.enable_grad():
                loss = closure()
        
        self.state['step'] += 1 

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue 

                grad = p.grad
                
                if grad.is_sparse:
                    raise RuntimeError("GaLore doesn't support sparse grads")
                if grad.ndim < 2: 
                    continue
                # --- END OF CORRECTION ---

                state = self.state[p]
                m, n = grad.shape
                effective_rank = min(group['rank'], m, n)

                if len(state) == 0:
                    state['projector'] = None 
                    state['exp_avg'] = torch.zeros((effective_rank, n), device=p.device)
                    state['exp_avg_sq'] = torch.zeros((effective_rank, n), device=p.device)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                is_update_step = (self.state['step'] - 1) % group['subspace_change_frequency'] == 0 
                projector = state.get('projector')

                if is_update_step or projector is None:
                    if group['svd_method'] == 'randomized':
                        U, _, _ = randomized_svd(grad, effective_rank)
                    else:
                        U, _, _ = torch.linalg.svd(grad, full_matrices=False)
                    
                    projector = U[:, :effective_rank]
                    state['projector'] = projector 

                projected_grad = projector.T @ grad
                beta1, beta2 = group['betas']
                
                exp_avg.mul_(beta1).add_(projected_grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(projected_grad, projected_grad.conj(), value=1 - beta2)

                bias_correction_1 = 1 - beta1 ** self.state['step']
                bias_correction_2 = 1 - beta2 ** self.state['step']

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction_2)).add_(group['eps']) 
                low_rank_update = (exp_avg / bias_correction_1).div_(denom)
                full_rank_update = projector @ low_rank_update

                if group['weight_decay'] != 0:
                    full_rank_update.add_(p, alpha=group['weight_decay'])

                p.add_(full_rank_update, alpha=-group['lr'])
                
        return loss


class BenchmarkModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 4096)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))

def calculate_optimizer_state_size(optimizer):
    total_size_bytes = 0
    for param_state in optimizer.state.values():
        if isinstance(param_state, dict):
            for state_tensor in param_state.values():
                if torch.is_tensor(state_tensor):
                    total_size_bytes += state_tensor.numel() * state_tensor.element_size()
    return total_size_bytes / (1024 * 1024)

def run_benchmark(optimizer_class, optimizer_name, device, opt_config):
    print(f"\n--- Benchmarking {optimizer_name} ---")
    
    model = BenchmarkModel().to(device)
    optimizer = optimizer_class(model.parameters(), **opt_config)
    
    dummy_input = torch.randn(64, 1024, device=device)
    dummy_labels = torch.randint(0, 10, (64,), device=device)
    criterion = nn.CrossEntropyLoss()

    for _ in range(10):
        optimizer.zero_grad()
        outputs = model(dummy_input)
        loss = criterion(outputs, dummy_labels)
        loss.backward()
        optimizer.step()

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    
    start_time = time.perf_counter()
    for i in range(100):
        optimizer.zero_grad()
        outputs = model(dummy_input)
        loss = criterion(outputs, dummy_labels)
        loss.backward()
        optimizer.step()
    
    if device == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    total_time = end_time - start_time
    peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device == "cuda" else 0
    theoretical_size_mb = calculate_optimizer_state_size(optimizer)

    print(f"Total Time (100 steps): {total_time:.4f} seconds")
    print(f"Optimizer State Size: {theoretical_size_mb:.2f} MB")
    if device == "cuda":
        print(f"Peak GPU Memory Allocated: {peak_memory_mb:.2f} MB")

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    if DEVICE == "cpu":
        print("benchmarking is only meaningful on CUDA.")

    adam_config = {'lr': 0.001}
    run_benchmark(torch.optim.Adam, "Standard Adam", DEVICE, adam_config)

    galore_exact_config = {
        'lr': 0.001, 
        'rank': 128, 
        'subspace_change_frequency': 50,
        'svd_method': 'exact'
    }
    run_benchmark(Galore, "Galore (Exact SVD)", DEVICE, galore_exact_config)

    galore_random_config = {
        'lr': 0.001, 
        'rank': 128, 
        'subspace_change_frequency': 50,
        'svd_method': 'randomized'
    }
    run_benchmark(Galore, "Galore (Randomized SVD)", DEVICE, galore_random_config)