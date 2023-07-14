# From cocob.py
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.optim.optimizer import Optimizer

class FREE_ADAGRAD(Optimizer):
    r"""Implements FREE_ADAGRAD algorithm.
    It has been proposed in `Parameter-free projected gradient descent`_.

    Remark: the paper provides bounds for the regret, only is the mean of the trajectory is used as the parameter value.

    See FREE_ADAGRAD_mean, FREE_ADAGRAD_sliding_mean or FREE_ADAGRAD_low_pass_filter to use example on how to get and use such a mean.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups

        initial_guess_of_distance_to_solution: initial guess of the distance between 
           the first value to the solution one (| x_1 - x_* | in the publication)

    .. _Parameter-free projected gradient descent
        https://arxiv.org/abs/2305.19605
    """

    def __init__(self, params, initial_guess_of_distance_to_solution: float = 1e-3):
        if initial_guess_of_distance_to_solution <= 0:
            raise ValueError(f"Invalid initial_guess_of_distance_to_solution value: {initial_guess_of_distance_to_solution}")

        super(FREE_ADAGRAD, self).__init__(params, dict(
            initial_guess_of_distance_to_solution = initial_guess_of_distance_to_solution
        ))

    @torch.no_grad()
    def step(self, closure = None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if len(self.param_groups) > 1:
            raise RuntimeError('FREE_ADAGRAD only supports 1 param group')

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state['gamma'] = torch.tensor(self.defaults["initial_guess_of_distance_to_solution"], device=p.data.device).detach()
                    state['Gamma'] = torch.tensor(0.0, device=p.data.device).detach()
                    state['x1'] = torch.clone(p.data).detach()
                    state['S'] = torch.tensor(0.0, device=p.data.device).detach()
                    state['k'] = torch.tensor(1.0, device=p.data.device).detach()

                # new x
                norm_g = torch.linalg.norm(p.grad)
                state['S'] += norm_g ** 2

                h = torch.sqrt((state['S'] + 1.0) * (1.0 + torch.log(1.0 + state['S'])))

                while True:
                    x_plus = p.data - state['gamma'] / h * p.grad
                    B = (2.0 / torch.sqrt(state['k'])) * state['gamma'] \
                        + torch.sqrt(state['Gamma'] + (state['gamma'] * norm_g / h) ** 2)
                    if torch.linalg.norm(x_plus - state['x1']) > B:
                        state['gamma'] *= 2
                        state['k'] += 1
                    else :
                        state['Gamma'] += (state['gamma'] * (norm_g / h)) ** 2
                    break

                # update model parameters
                p.data.copy_(x_plus)

        return loss

class FREE_ADAGRAD_mean:
    r"""Class that helps calculate the average of parameters (as a function of iteration).
    The mean is computed online (only the previous value is stored, not the whole history)
    Example of use:
        optimizer = opt_func(parameters)
        fam = FREE_ADAGRAD_mean()
        for i in range( 25 ):
            # usual stuff
            loss = cost_func(parameters)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # register value
            fam.add(parameters)

        # use the mean
        fam.set(parameters)
    """

    def __init__(self):
        self.output = None
        self.n = 0

    def add(self, params):
        """ register parameter value """ 

        if self.output is None:
            self.output = [torch.clone(p.data) for p in params]
        else:
            ind = 0
            for p in params:
                self.output[ind] = (self.output[ind] * self.n + p.data) / (self.n + 1)
                ind += 1
                
        self.n += 1

    def set(self, params):
        """ assign output (the mean) to parameters """ 
        
        ind = 0
        for p in params:
            p.data = self.output[ind]
            ind += 1

class FREE_ADAGRAD_low_pass_filter:
    r"""Class that helps calculate a (first order recursive) low pass filter of the parameters (as a function of #iteration).
        output = (1 - self.coeff) * output + self.coeff * parameters
    Arguments:
        coeff (float): how much the new parameter value is taken into account at each new iteration
    Example of use:
        optimizer = opt_func(parameters)
        fam = FREE_ADAGRAD_low_pass_filter(0.9)
        for i in range( 25 ):
            # usual stuff
            loss = cost_func(parameters)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # register value
            fam.add(parameters)

        # use the output
        fam.set(parameters)
    """
    
    def __init__(self, coeff : float = 0.5):
        self.coeff = coeff
        self.output = None

    def add(self, params):
        """ register parameter value """ 

        if self.output is None:
            self.output = [torch.clone(p.data) for p in params]
        else:
            ind = 0
            for p in params:
                self.output[ind] = (1 - self.coeff) * self.output[ind] + self.coeff * p.data
                ind += 1

    def set(self, params):
        """ assign output (the low pass filter) to parameters """ 
        
        ind = 0
        for p in params:
            p.data = self.output[ind]
            ind += 1

class FREE_ADAGRAD_sliding_mean:
    r"""Class that helps calculate a sliding mean of the parameters (as a function of iteration).
    Arguments:
        max_n (int): width of the window
    Example of use:
        optimizer = opt_func(parameters)
        fam = FREE_ADAGRAD_sliding_mean(10)
        for i in range( 25 ):
            # usual stuff
            loss = cost_func(parameters)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # register value
            fam.add(parameters)

        # use the output
        fam.set(parameters)
    """
    
    def __init__(self, max_n: int):
        self.max_n = max_n
        self.outputs = []

    def add(self, params):
        """ register parameter value """ 

        self.outputs.append([torch.clone(p.data) for p in params])
        while len(self.outputs) > self.max_n:
            self.outputs.pop(0)

    def set(self, params):
        """ assign output (the sliding mean) to parameters """ 

        ind = 0        
        no = len(self.outputs)
        for p in params:
            p.data = sum([output[ind] for output in self.outputs]) / no
            ind += 1

