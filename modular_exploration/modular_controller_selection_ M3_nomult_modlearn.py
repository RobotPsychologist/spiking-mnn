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
        if t % 10 < 3.3:
            return [0,1,1]
        elif t % 10 > 3.3 and t % 10 < 6.6:
            return [1,0,1]
        else:
            return [1,1,0]
            
    def stim_func(t):
        '''The changing signal that we want the network to learn.'''
        if t % 10 < 3.3:
            return np.sin(t*2*np.pi*2)
        elif t % 10 > 3.3 and t % 10 < 6.6:
            return 1.5
        else:
            return -1.5

    def out1_func(t):
        '''The output signal that we want the network to learn.'''
        if t % 10 < 3.3:
            return np.sin(t*2*np.pi*2)**2
        elif t % 10 > 3.3 and t % 10 < 6.6:
            return 0
        else:
            return 0
    
    def out2_func(t):
        '''The output signal that we want the network to learn.'''
        if t % 10 < 3.3:
            return 0
        elif t % 10 > 3.3 and t % 10 < 6.6:
            return 1.5**2
        else:
            return 0

    def out3_func(t):
        '''The output signal that we want the network to learn.'''
        if t % 10 < 3.3:
            return 0
        elif t % 10 > 3.3 and t % 10 < 6.6:
            return 0
        else:
            return -1.5*2 
    def mult(x):
        '''Multiplies together the stimulus and the result from the controller.'''
        return x[0]*x[1]

    def init_func(x):
        return [0,0,0]

    def mod_init_func(x):
        return [0]  
    # Nodes
    off_learning_node = nengo.Node(0, size_out=1, label='off_learning_node') # Provides us with the ability to turn off learning to see if the network remembers how to alternate between modules.
    gt_ctrl_node = nengo.Node(gt_ctrl, size_out=3, label='gt_ctrl_node') # Provides the ground truth to the controller for learning.
    gt_mod1_out_node = nengo.Node(out1_func, size_out=1, label='gt_mod1_out_node')
    gt_mod2_out_node = nengo.Node(out2_func, size_out=1, label='gt_mod2_out_node')
    gt_mod3_out_node = nengo.Node(out3_func, size_out=1, label='gt_mod3_out_node')
    stim_node = nengo.Node(stim_func, size_out=1, label='stim_node') # NETWORK INPUT
    #output_node = nengo.Node(None, size_in=1, label='output_node') # NETWORK OUTPUT

    # LDN Set Up
    ldn = nengo.Ensemble(n_neurons=400, dimensions=q, label='ldn')
    nengo.Connection(stim_node, ldn, transform=B, synapse=tau_rec)
    nengo.Connection(ldn, ldn, transform=A, synapse=tau_rec) # Recurrent connection.

    # Ensembles 
    error_controller = nengo.Ensemble(n_neurons=100, dimensions=3, label='error_controller') # Used for training the connection between the LDN and controller
    error_mod1_out = nengo.Ensemble(n_neurons=100, dimensions=1, label='error_mod1_out', radius=2) 
    error_mod2_out = nengo.Ensemble(n_neurons=100, dimensions=1, label='error_mod2_out', radius=2) 
    error_mod3_out = nengo.Ensemble(n_neurons=100, dimensions=1, label='error_mod3_out', radius=2) 
    controller = nengo.Ensemble(n_neurons=100, dimensions=3, label='controller', radius=np.sqrt(2)) # The conroller should learn to send a [1,0] or [0,1] depending on the signal it is recieving.
    m1 = nengo.Ensemble(n_neurons=100, dimensions=1, radius=2, label='m1') # The module that is suppose to learn only learn the sinusoidal signal
    m2 = nengo.Ensemble(n_neurons=100, dimensions=1, radius=2, label='m2') # The module that is supposed to learn the constant signal.
    m3 = nengo.Ensemble(n_neurons=100, dimensions=1, radius=2, label='m3') # The module that is supposed to learn the constant signal.
    output_ens = nengo.Ensemble(n_neurons=100, dimensions=1, radius=5, label='output_node') # NETWORK OUTPUT
    # Connections
    ## INPUTS
    nengo.Connection(off_learning_node, error_controller.neurons,transform=10*np.ones([100,1])) # To control learning, turns learning off when very negative.
    nengo.Connection(stim_node, m1) # Connects the stimulus that multi will combine with the input from controller
    nengo.Connection(stim_node, m2) # Connects the stimulus that multi will combine with the input from controller
    nengo.Connection(stim_node, m3) # Connects the stimulus that multi will combine with the input from controller
    nengo.Connection(gt_ctrl_node, error_controller, transform=-1) # Connects the ground truth stimulus that the controller error population.
    
    nengo.Connection(gt_mod1_out_node, error_mod1_out, transform=-1) 
    nengo.Connection(gt_mod2_out_node, error_mod2_out, transform=-1) 
    nengo.Connection(gt_mod3_out_node, error_mod3_out, transform=-1) 
    
    ## NETWORK CONNECTIONS
    ldn_ctrl0 = nengo.Connection(ldn, controller,synapse=prb_syn,function=init_func, learning_rule_type=nengo.PES(learning_rate=1e-03)) # The memory of the system so that controller can learn the difference between sin and constant.
    controller_mult1_conn = nengo.Connection(controller[0], m1.neurons, transform=-10*np.ones([100,1])) 
    controller_mult2_conn = nengo.Connection(controller[1], m2.neurons, transform=-10*np.ones([100,1])) 
    controller_mult3_conn = nengo.Connection(controller[2], m3.neurons, transform=-10*np.ones([100,1])) 

    ## OUTPUTS
    m1_output = nengo.Connection(m1, output_ens,synapse=prb_syn,function=mod_init_func, learning_rule_type=nengo.PES(learning_rate=1e-03)) 
    m2_output = nengo.Connection(m2, output_ens,synapse=prb_syn,function=mod_init_func, learning_rule_type=nengo.PES(learning_rate=1e-03)) 
    m3_output = nengo.Connection(m3, output_ens,synapse=prb_syn,function=mod_init_func, learning_rule_type=nengo.PES(learning_rate=1e-03)) 

    ## ERROR CONNECTIONS
    nengo.Connection(controller, error_controller, transform=1) # Connects the controller output to the error population.
    nengo.Connection(error_controller, ldn_ctrl0.learning_rule)
    
    nengo.Connection(output_ens, error_mod1_out, transform=1)
    nengo.Connection(output_ens, error_mod2_out, transform=1)
    nengo.Connection(output_ens, error_mod3_out, transform=1)
    nengo.Connection(error_mod1_out, m1_output.learning_rule, transform=1)
    nengo.Connection(error_mod2_out, m2_output.learning_rule, transform=1)
    nengo.Connection(error_mod3_out, m3_output.learning_rule, transform=1) 