import nengo
import numpy as np
from scipy.special import legendre

# n_neurons: num of neurons per memory vector
# theta: length of time window (in seconds)
# q: LMU memory vector size
# size_in: dim of signal to remember 
 # Use:
theta = 1
q=4
n_neurons=400
tau=0.03
prb_syn=0.01
tau_rec   = 100/1000

A = np.zeros((q, q))
B = np.zeros((q, 1))
for i in range(q):
    B[i] = (-1.)**i * (2*i+1)
    for j in range(q):
        A[i,j] = (2*i+1)*(-1 if i<j else (-1.)**(i-j+1))
A = A / theta
B = B / theta
A = tau_rec * A + np.eye(q)
B = tau_rec * B

model = nengo.Network()

with model:
    def stim_func(t):
        if t % 10 < 5:
            #return np.sin(t*2*np.pi*2)
            return 2
        else:
            return -0.5
    
    def gt_ctrl(t):
        if t % 10 < 5:
            return [1,0]
        else:
            return [0,1]
    
    def init_func(x):
        return [0,0]
    
    def mult(x):
        return x[0]*x[1]
    
    def ideal_func(x):
        return x**2
    
    # Nodes
    gt_crl_node = nengo.Node(gt_ctrl, size_out=2)
    stim = nengo.Node(stim_func, size_out=1)
    output = nengo.Node(None, size_in=1)   
    off_learning = nengo.Node(0, size_out=1)
    
    #LMU
    ldn = nengo.Ensemble(n_neurons=400, dimensions=q)
    nengo.Connection(stim, ldn, transform=B, synapse=tau_rec)
    nengo.Connection(ldn, ldn, transform=A, synapse=tau_rec)

    # Ensembles 
    controller = nengo.Ensemble(n_neurons=100, dimensions=2) 
    error_controller = nengo.Ensemble(n_neurons=100, dimensions=2)
    mult1 = nengo.Ensemble(n_neurons=100, dimensions=2, radius=2)
    mult2 = nengo.Ensemble(n_neurons=100, dimensions=2, radius=2) 
    #m1 = nengo.Ensemble(n_neurons=100, dimensions=1)
    #m2 = nengo.Ensemble(n_neurons=100, dimensions=1)
    m1 = nengo.Node(lambda t, x: np.sin(x)**2, size_in=1)
    m2 = nengo.Node(lambda t, x: x**2, size_in=1)

    # Connections
    nengo.Connection(off_learning, error_controller.neurons,transform=10*np.ones([100,1]))
    ## Network Connections
    #nengo.Connection(stim, controller[0])
    #nengo.Connection(stim, controller[1])
    nengo.Connection(stim, mult1[1])
    nengo.Connection(stim, mult2[1])
    nengo.Connection(gt_crl_node, error_controller, transform=-1)
    nengo.Connection(controller, error_controller, transform=1)
    ldn_ctrl0 = nengo.Connection(ldn, controller,synapse=prb_syn,function=init_func, learning_rule_type=nengo.PES(learning_rate=1e-05))
    
    #ldn_ctrl1 = nengo.Connection(ldn.lmu.ea_ensembles[0], controller[1],synapse=prb_syn,learning_rule_type=nengo.PES())
    
    controller_mult1_conn = nengo.Connection(controller[0], mult1[0])
    controller_mult2_conn = nengo.Connection(controller[1], mult2[0])
    mult1_m1_conn = nengo.Connection(mult1, m1, function=mult)
    mult2_m2_conn = nengo.Connection(mult2, m2, function=mult)
    
    ## Error Connections
    nengo.Connection(error_controller, ldn_ctrl0.learning_rule)
    #nengo.Connection(error_controller[1], ldn_ctrl1.learning_rule)
    
    #nengo.Connection(error_controller, controller)
    
    m1_output_conn = nengo.Connection(m1, output) #, function=init_func, learning_rule_type=nengo.PES(learning_rate=1e-6))
    m2_output_conn = nengo.Connection(m2, output) #, function=init_func, learning_rule_type=nengo.PES(learning_rate=1e-6))
