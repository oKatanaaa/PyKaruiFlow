import numpy as np
from .optimizer import Optimizer, required
from pykaruiflow.core import Parameter, tensor


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(Adam, self).__init__(params, defaults)
        self.__init_state()

    def __init_state(self):
        for group in self.param_groups:
            state = group['state'] = {}
            for p in group['params']:
                if p.requires_grad:
                    state[p] = {
                        'step': 0,
                        'exp_avg': Parameter(np.zeros_like(p.data)),  # m_t
                        'exp_avg_sq': Parameter(np.zeros_like(p.data))  # v_t
                    }

    def step(self):
        """Performs a single optimization step."""
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            state = group['state']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Retrieve state
                exp_avg = state[p]['exp_avg']  # m_t
                exp_avg_sq = state[p]['exp_avg_sq']  # v_t
                step = state[p]['step'] + 1
                state[p]['step'] = step

                # Update biased first moment estimate
                exp_avg = exp_avg * beta1 + tensor(p.grad) * (1 - beta1)

                # Update biased second moment estimate
                exp_avg_sq = exp_avg_sq * beta2 + tensor(p.grad ** 2) * (1 - beta2)

                # Compute bias-corrected first moment estimate
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = lr / bias_correction1

                # Compute bias-corrected second moment estimate
                denom = (np.sqrt(exp_avg_sq.data)/ (bias_correction2 ** 0.5)) + eps

                # Update parameters
                p += tensor(exp_avg.data / denom) * -step_size

                # Assign the updated moments back to state
                state[p]['exp_avg'] = exp_avg
                state[p]['exp_avg_sq'] = exp_avg_sq

        return None
