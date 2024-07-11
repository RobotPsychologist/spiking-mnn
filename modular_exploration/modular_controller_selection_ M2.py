import nengo
import numpy as np
from scipy.special import legendre
# LMU Set Up
theta = 1 #theta: length of time window (in seconds)
q=4 # q: LMU memory vector size
n_neurons=400 #n_neurons: num of neurons per memory vector
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

# Network 
model = nengo.Network()

with model:
    def gt_ctrl(t):
        '''The provides the ground truth to the controller for which on/off
        signal should be sent to mult1 and mult2.
        '''
        if t % 10 < 5:
            return [1,0]
        else:
            return [0,1]
            
    def stim_func(t):
        '''The changing signal that we want the network to learn.'''
        if t % 10 < 5:
            return np.sin(t*2*np.pi*2)
        else:
            return -0.5
    
    def mult(x):
        '''Multiplies together the stimulus and the result from the controller.'''
        return x[0]*x[1]

    def init_func(x):
        return [0,0]
    
    # Nodes
    off_learning_node = nengo.Node(0, size_out=1) # Provides us with the ability to turn off learning to see if the network remembers how to alternate between modules.
    gt_crl_node = nengo.Node(gt_ctrl, size_out=2) # Provides the ground truth to the controller for learning.
    stim_node = nengo.Node(stim_func, size_out=1) # NETWORK INPUT
    output_node = nengo.Node(None, size_in=1) # NETWORK OUTPUT

    # LDN Set Up
    ldn = nengo.Ensemble(n_neurons=400, dimensions=q)
    nengo.Connection(stim_node, ldn, transform=B, synapse=tau_rec)
    nengo.Connection(ldn, ldn, transform=A, synapse=tau_rec) # Recurrent connection.

    # Ensembles 
    error_controller = nengo.Ensemble(n_neurons=100, dimensions=2) # Used for training the connection between the LDN and controller
    
    controller = nengo.Ensemble(n_neurons=100, dimensions=2) # The conroller should learn to send a [1,0] or [0,1] depending on the signal it is recieving.
    mult1 = nengo.Ensemble(n_neurons=100, dimensions=2, radius=2) # The ensemble that combines stim and controller
    mult2 = nengo.Ensemble(n_neurons=100, dimensions=2, radius=2) # The ensemble that combines stim and controller
    m1 = nengo.Ensemble(n_neurons=100, dimensions=1, radius=2) # The module that is suppose to learn only learn the sinusoidal signal
    m2 = nengo.Ensemble(n_neurons=100, dimensions=1, radius=2) # The module that is supposed to learn the constant signal.

    # Connections
    ## INPUTS
    nengo.Connection(off_learning_node, error_controller.neurons,transform=10*np.ones([100,1])) # To control learning, turns learning off when very negative.
    nengo.Connection(stim_node, mult1[1]) # Connects the stimulus that multi will combine with the input from controller
    nengo.Connection(stim_node, mult2[1]) # Connects the stimulus that multi will combine with the input from controller
    nengo.Connection(gt_crl_node, error_controller, transform=-1) # Connects the ground truth stimulus that the controller error population.
    
    ## NETWORK CONNECTIONS
    ldn_ctrl0 = nengo.Connection(ldn, controller,synapse=prb_syn,function=init_func, learning_rule_type=nengo.PES(learning_rate=1e-05)) # The memory of the system so that controller can learn the difference between sin and constant.
    controller_mult1_conn = nengo.Connection(controller[0], mult1[0])
    controller_mult2_conn = nengo.Connection(controller[1], mult2[0])    
    mult1_m1_conn = nengo.Connection(mult1, m1, function=mult)
    mult2_m2_conn = nengo.Connection(mult2, m2, function=mult)

    ## OUTPUTS
    m1_output_conn = nengo.Connection(m1, output_node) #, function=init_func, learning_rule_type=nengo.PES(learning_rate=1e-6))
    m2_output_conn = nengo.Connection(m2, output_node) #, function=init_func, learning_rule_type=nengo.PES(learning_rate=1e-6))
    
    ## ERROR CONNECTIONS
    nengo.Connection(controller, error_controller, transform=1) # Connects the controller output to the error population.
    nengo.Connection(error_controller, ldn_ctrl0.learning_rule)
