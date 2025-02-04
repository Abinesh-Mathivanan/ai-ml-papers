import numpy as np

def adam_optimizer(params, grads, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    if not hasattr(adam_optimizer, "m"):
        adam_optimizer.m = {key: np.zeros_like(value) for key, value in params.items()}
        adam_optimizer.v = {key: np.zeros_like(value) for key, value in params.items()}
        adam_optimizer.t = 0
    
    adam_optimizer.t += 1
    
    for key in params.keys():
        adam_optimizer.m[key] = beta1 * adam_optimizer.m[key] + (1 - beta1) * grads[key]
        adam_optimizer.v[key] = beta2 * adam_optimizer.v[key] + (1 - beta2) * np.square(grads[key])
        
        m_hat = adam_optimizer.m[key] / (1 - beta1 ** adam_optimizer.t)
        v_hat = adam_optimizer.v[key] / (1 - beta2 ** adam_optimizer.t)
        
        params[key] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    
    return params