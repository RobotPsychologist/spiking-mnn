import nengo
import numpy as np

model = nengo.Network()
with model:
    
    def stim_func(t):
        if t % 10 < 5:
            return np.sin(t*2*np.pi*2)
        else:
            return -0.5
    stim = nengo.Node(stim_func)
    

    controller = nengo.Ensemble(n_neurons=100, dimensions=1)
    
    m1 = nengo.Ensemble(n_neurons=100, dimensions=1)
    m2 = nengo.Ensemble(n_neurons=100, dimensions=1)
    
    nengo.Connection(stim, controller)
    
    def init_func(x):
        return 0.5
        
    conn_cont_mod1 = nengo.Connection(controller, m1, function=init_func, learning_rule_type=nengo.PES(learning_rate=1e-4))
    conn_cont_mod2 = nengo.Connection(controller, m2, function=init_func, learning_rule_type=nengo.PES(learning_rate=1e-4))

    output = nengo.Node(None, size_in=1)
    

    conn1 = nengo.Connection(m1, output, function=init_func, learning_rule_type=nengo.PES(learning_rate=1e-6))
    conn2 = nengo.Connection(m2, output, function=init_func, learning_rule_type=nengo.PES(learning_rate=1e-6))
    
    
    error_mod1 = nengo.Node(None, size_in=1)
    error_mod2 = nengo.Node(None, size_in=1)
    error_cont = nengo.Node(None, size_in=1)
    
    nengo.Connection(output, error_mod1)
    nengo.Connection(output, error_mod2)
    nengo.Connection(output, error_cont)
    
    
    def ideal_func(x):
        return x**2
    
    nengo.Connection(stim, error_mod1, function=ideal_func, transform=-1)
    nengo.Connection(stim, error_mod2, function=ideal_func, transform=-1)
    nengo.Connection(stim, error_cont, function=ideal_func, transform=-1)
    
    
    nengo.Connection(error_mod1, conn1.learning_rule)
    nengo.Connection(error_mod2, conn2.learning_rule)
    
    nengo.Connection(error_cont, conn_cont_mod1.learning_rule)
    nengo.Connection(error_cont, conn_cont_mod2.learning_rule)
