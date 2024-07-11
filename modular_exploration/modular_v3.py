import nengo
import numpy as np

import numpy 
import nengo
from scipy.special import legendre

# n_neurons: num of neurons per memory vector
# theta: length of time window (in seconds)
# q: LMU memory vector size
# size_in: dim of signal to remember 
class LMUNetwork(nengo.Network):
    def __init__(self, n_neurons, theta, q, size_in=1, tau=0.05,**kwargs):
        super().__init__()
        
        self.q = q              # number of internal state dimensions per input
        self.theta = theta      # size of time window (in seconds)
        self.size_in = size_in  # number of inputs

        # Do Aaron's math to generate the matrices
        #  https://github.com/arvoelke/nengolib/blob/master/nengolib/synapses/analog.py#L536
        Q = np.arange(q, dtype=np.float64)
        R = (2*Q + 1)[:, None] / theta
        j, i = np.meshgrid(Q, Q)

        self.A = np.where(i < j, -1, (-1.)**(i-j+1)) * R
        self.B = (-1.)**Q[:, None] * R
        
        with self:
            self.input = nengo.Node(size_in=size_in)
            self.reset = nengo.Node(size_in=1)
            
            self.lmu = nengo.networks.EnsembleArray(n_neurons, n_ensembles=size_in, 
                                                    ens_dimensions=q, **kwargs)
            self.output = self.lmu.output            
            for i in range(size_in):
                nengo.Connection(self.input[i], self.lmu.ea_ensembles[i], synapse=tau,
                                 transform = tau*self.B)
                nengo.Connection(self.lmu.ea_ensembles[i], self.lmu.ea_ensembles[i], synapse=tau,
                                 transform = tau*self.A + np.eye(q))
                nengo.Connection(self.reset, self.lmu.ea_ensembles[i].neurons, transform = [[-2.5]]*n_neurons, synapse=None)


theta = 0.2
q=10
n_neurons=800
tau=0.03
prb_syn=0.01

model = nengo.Network()
with model:
    def stim_func(t):
        if t % 10 < 5:
            return np.sin(t*2*np.pi*2)
        else:
            return -0.5
    
    def gt_ctrl(t):
        if t % 10 < 5:
            return [1,0]
        else:
            return [0,1]
    
    def init_func(x):
        return 0.5
    
    def mult(x):
        return x[0]*x[1]
    
    def ideal_func(x):
        return x**2
    
    # Nodes
    gt_crl_node = nengo.Node(gt_ctrl, size_out=2)
    ldn = LMUNetwork(n_neurons=n_neurons, theta=theta, q=q, size_in=1, tau=tau)
    stim = nengo.Node(stim_func, size_out=1)
    output = nengo.Node(None, size_in=1)   
    
    # Ensembles 
    controller = nengo.Ensemble(n_neurons=100, dimensions=2)
    pre_controller = nengo.Ensemble(n_neurons=100, dimensions=2)
    
    mult1 = nengo.Ensemble(n_neurons=100, dimensions=2)
    mult2 = nengo.Ensemble(n_neurons=100, dimensions=2)
    #m1 = nengo.Ensemble(n_neurons=100, dimensions=1)
    #m2 = nengo.Ensemble(n_neurons=100, dimensions=1)
    m1 = nengo.Node(lambda t, x: np.sin(x)**2, size_in=1)
    m2 = nengo.Node(lambda t, x: x**2, size_in=1)
    
    ## Errors
    error_controller = nengo.Ensemble(n_neurons=1000, dimensions=2)
    
    # Connections
    ## Network Connections
    #nengo.Connection(stim, controller[0] )
    #nengo.Connection(stim, controller[1] )
    nengo.Connection(stim, pre_controller[0] )
    nengo.Connection(stim, pre_controller[1] )
    precon_cont_conn1 = nengo.Connection(pre_controller[0], controller[0], learning_rule_type=nengo.PES() )
    precon_cont_conn2 = nengo.Connection(pre_controller[1], controller[1], learning_rule_type=nengo.PES() )
    
    nengo.Connection(stim, mult1[1])
    nengo.Connection(stim, mult2[1])
    nengo.Connection(gt_crl_node, error_controller)
    
    controller_mult1_conn = nengo.Connection(controller[0], mult1[0], learning_rule_type=nengo.PES() )
    controller_mult2_conn = nengo.Connection(controller[1], mult2[0], learning_rule_type=nengo.PES() )
    mult1_m1_conn = nengo.Connection(mult1, m1, function=mult)
    mult2_m2_conn = nengo.Connection(mult2, m2, function=mult)
    
    ## Error Connections
    nengo.Connection(error_controller[0], precon_cont_conn1.learning_rule)
    nengo.Connection(error_controller[1], precon_cont_conn2.learning_rule)
    
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
