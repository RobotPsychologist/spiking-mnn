import numpy as np
#import matplotlib.pyplot as plt

#%matplotlib inline

import nengo
from nengo.processes import WhiteSignal

model = nengo.Network()
with model:
    # -- input and pre popluation
    inp = nengo.Node(WhiteSignal(60, high=5), size_out=2)
    pre = nengo.Ensemble(120, dimensions=2)
    nengo.Connection(inp, pre)

    # -- post populations
    post_pes = nengo.Ensemble(60, dimensions=1)
    post_rls = nengo.Ensemble(60, dimensions=1)

    # -- reference population, containing the actual product
    product = nengo.Ensemble(60, dimensions=1)
    nengo.Connection(inp, product, function=lambda x: x[0] * x[1], synapse=None)

    # -- error populations
    error_pes = nengo.Ensemble(60, dimensions=1)
    nengo.Connection(post_pes, error_pes)
    nengo.Connection(product, error_pes, transform=-1)
    error_rls = nengo.Ensemble(60, dimensions=1)
    nengo.Connection(post_rls, error_rls)
    nengo.Connection(product, error_rls, transform=-1)

    # -- learning connections
    conn_pes = nengo.Connection(
        pre,
        post_pes,
        function=lambda x: np.random.random(1),
        learning_rule_type=nengo.PES(),
    )
    nengo.Connection(error_pes, conn_pes.learning_rule)
    conn_rls = nengo.Connection(
        pre,
        post_rls,
        function=lambda x: np.random.random(1),
        learning_rule_type=nengo.RLS(),
    )
    nengo.Connection(error_rls, conn_rls.learning_rule)

    # -- inhibit errors after 40 seconds
    inhib = nengo.Node(lambda t: 2.0 if t > 40.0 else 0.0)
    nengo.Connection(inhib, error_pes.neurons, transform=[[-1]] * error_pes.n_neurons)
    nengo.Connection(inhib, error_rls.neurons, transform=[[-1]] * error_rls.n_neurons)

    # -- probes
    #product_p = nengo.Probe(product, synapse=0.01)
    #pre_p = nengo.Probe(pre, synapse=0.01)
    #post_pes_p = nengo.Probe(post_pes, synapse=0.01)
    #error_pes_p = nengo.Probe(error_pes, synapse=0.03)
    #post_rls_p = nengo.Probe(post_rls, synapse=0.01)
    #error_rls_p = nengo.Probe(error_rls, synapse=0.03)

#with nengo.Simulator(model) as sim:
  #  sim.run(60)
