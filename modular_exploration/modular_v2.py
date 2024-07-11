import nengo
import numpy as np

model = nengo.Network()
with model:
    def stim_func(t):
        if t % 10 < 5:
            return np.sin(t*2*np.pi*2)
        else:
            return -0.5
    
    def gt_ctrl(t):
        if t % 10 < 5:
            return 1
        else:
            return 0
    
    def init_func(x):
        return 0.5
    
    def mult(x):
        return x[0]*x[1]
    
    def ideal_func(x):
        return x**2
    
    # Nodes
    gt_crl_node = nengo.Node(gt_ctrl, size_out=1)
    stim = nengo.Node(stim_func, size_out=1)
    output = nengo.Node(None, size_in=1)   
    
    # Ensembles 
    controller = nengo.Ensemble(n_neurons=100, dimensions=2)
    mult1 = nengo.Ensemble(n_neurons=100, dimensions=2)
    mult2 = nengo.Ensemble(n_neurons=100, dimensions=2)
    #m1 = nengo.Ensemble(n_neurons=100, dimensions=1)
    #m2 = nengo.Ensemble(n_neurons=100, dimensions=1)
    m1 = nengo.Node(lambda t, x: np.sin(x)**2, size_in=1)
    m2 = nengo.Node(lambda t, x: x**2, size_in=1)
    
    ## Errors
    error_controller = nengo.Ensemble(n_neurons=100, dimensions=2)
    
    # Connections
    ## Network Connections
    nengo.Connection(stim, controller[0])
    nengo.Connection(stim, controller[1])
    nengo.Connection(stim, mult1[1])
    nengo.Connection(stim, mult2[1])
    nengo.Connection(gt_crl_node, error_controller[0])
    
    controller_mult1_conn = nengo.Connection(controller[0], mult1[0], learning_rule_type=nengo.PES() )
    controller_mult2_conn = nengo.Connection(controller[1], mult2[0], learning_rule_type=nengo.PES() )
    mult1_m1_conn = nengo.Connection(mult1, m1, function=mult)
    mult2_m2_conn = nengo.Connection(mult2, m2, function=mult)
    
    ## Error Connections
    nengo.Connection(error_controller[0], controller_mult1_conn.learning_rule)
    nengo.Connection(error_controller[1], controller_mult2_conn.learning_rule)
    
    #nengo.Connection(error_controller, controller)
    
    m1_output_conn = nengo.Connection(m1, output) #, function=init_func, learning_rule_type=nengo.PES(learning_rate=1e-6))
    m2_output_conn = nengo.Connection(m2, output) #, function=init_func, learning_rule_type=nengo.PES(learning_rate=1e-6))
    
    #nengo.Connection(output, error_controller)
    
    stop_learning=nengo.Node(0)
    

    #nengo.Connection(stim, error_mod1, function=ideal_func, transform=-1)
    #nengo.Connection(stim, error_mod2, function=ideal_func, transform=-1)
    #nengo.Connection(stim, error_cont, function=ideal_func, transform=-1)
    
    
    #nengo.Connection(error_mod1, conn1.learning_rule)
    #nengo.Connection(error_mod2, conn2.learning_rule)
    
    #nengo.Connection(error_cont, conn_cont_mod1.learning_rule)
    #nengo.Connection(error_cont, conn_cont_mod2.learning_rule)
