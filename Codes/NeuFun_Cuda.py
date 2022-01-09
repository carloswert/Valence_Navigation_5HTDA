import numpy as np
from numba import jit_module
"""
Different convolutions required for SRM0 model and the calculation of
the synaptic weight
"""

def convolution(conv_decay,conv_rise,tau_m,tau_s,eps0,X,w=None):
    """
    From SRM0 model
    Input: conv_decay = convolution, decay compontent
    conv_rise = convolution, rise component
    tau_m = decay time
    tau_s = rise time constant
    eps0 = multiplying constant
    X = spike train vector
    w = multiplying constant for X (weights of the input)
    Output: conv = total_convolution
    conv_decay = updated decay
    conv_rise = updated rise
    """
    if w is None:
        w = np.ones(X.shape).T
    conv_decay = conv_decay + (-conv_decay)/tau_m + np.multiply(X,w.T)
    conv_rise = conv_rise + (-conv_rise)/tau_s + np.multiply(X,w.T)
    conv = eps0*(conv_decay-conv_rise)/(tau_m-tau_s)
    return conv,conv_decay,conv_rise


def convolution_type2(conv1, tau_m, eps0, X):
    """
    exponential decay
    input =       conv1= convolution, decay component
                   tau_m = decay time constant
                   eps0= multiplying constant, scales the whole convolution
                   X = spike train vector, if there is a spike the convolution jumps by 1
    output =       conv = total convolution (scaled by eps0)
                   conv1 = convolution, decay component (updated)
    """
    conv1 = conv1 + (-conv1)/tau_m + X
    conv = np.multiply(eps0,(conv1))
    return conv, conv1

def weights_update_stdp(A_plus, A_minus, tau_plus, tau_minus, X, Y, conv1_pre, conv1_post, trace, tau_e):
    conv_pre_old, _ = convolution_type2(conv1_pre, tau_plus,  A_plus, np.zeros_like(conv1_pre)) #Pre trace without spike - for coincident spikes
    conv_post_old, _ = convolution_type2(conv1_post, tau_minus,A_minus, np.zeros_like(conv1_pre)) #Post trace without spike - for coincident spikes

    #STDP
    conv_pre, conv1_pre = convolution_type2(conv1_pre, tau_plus,  A_plus, X) #Trace by pre-synaptic neuron, amplitude A+ and time
    conv_post, conv1_post = convolution_type2(conv1_post, tau_minus,  A_minus, Y) #Trace by pre-synaptic neuron, amplitude A+ and time
    W = np.multiply((np.multiply(conv_pre,Y)+np.multiply(conv_post,X)),(X+Y!=2))+ (np.multiply(conv_pre_old,Y)+np.multiply(conv_post_old,X))+np.multiply((A_plus+A_minus)/2,(X+Y==2))
    #Elegibility trace
    tot_conv, trace = convolution_type2(trace, tau_e,  1, W) #All weight changes filtered through trace
    return conv1_pre, conv1_post,tot_conv, trace, W

jit_module(nopython=True, error_model="numpy")

def neuron(epsp, chi, last_spike_post, tau_m, rho0, theta, delta_u, i):
    N_pc, N_action = epsp.shape #no. place cells, no. action neurons
    u = np.sum(epsp,axis=0,keepdims=True).T+chi*np.exp((-i+last_spike_post)/tau_m) #membrane potential
    rho_tilda= rho0*np.exp((u-theta)/delta_u) #instanteneous rate
    Y= np.random.rand(N_action,1)<= rho_tilda #realization spike train
    last_spike_post[Y]=i #update time postsyn spike
    Canc = 1-np.matlib.repmat(Y, 1, N_pc) #1 if postsyn neuron spiked, 0 otherwise

    return Y, last_spike_post, Canc,u

def weights_update_rate(A, tau_STDP, r_X, r_Y, W, trace, tau_e):
    #Rate-based rule based on near neighboors STDP to BCM
    W = A*r_Y*(1/(tau_STDP**(-1)+r_Y))*r_X
    #Elegibility trace
    tot_conv, trace = convolution_type2(trace, tau_e,  1, W) #All weight changes filtered through trace
    return W, tot_conv, trace, W
