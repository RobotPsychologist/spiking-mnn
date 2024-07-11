import nengo
import numpy as np

model = nengo.Network()
with model:
    
    def stim_func(t):
        return np.sin(t*2*np.pi*2)
    stim = nengo.Node(stim_func)
    

    module = nengo.Ensemble(n_neurons=100, dimensions=1)
    
    nengo.Connection(stim, module)
    
    output = nengo.Node(None, size_in=1)
    
    def init_func(x):
        return 0.5
    conn = nengo.Connection(module, output, function=init_func, learning_rule_type=nengo.PES(learning_rate=1e-5))
    
    error = nengo.Node(None, size_in=1)
    nengo.Connection(output, error)
    
    
    
    def ideal_func(x):
        return x**2
    
    nengo.Connection(stim, error, function=ideal_func, transform=-1)
    
    nengo.Connection(error, conn.learning_rule)
    
    
