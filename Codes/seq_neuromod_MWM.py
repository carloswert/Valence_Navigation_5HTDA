from optparse import OptionParser
from numba import jit, cuda
import numpy as np
import numpy.matlib
import itertools
import time
from NeuFun_Cuda import convolution, convolution_type2, neuron, weights_update_stdp
from bisect import bisect
import multiprocessing
import pickle
import psutil

"""
Sequential weight change (SWC) implementation with a Delta function as 5-HT.

Options:
    - Output (-o or --output) takes a string with the name of the job.
    - Episodes (-e or --episodes) establishes the number of consecutive simulations of a trial.
    - Trials (-t or --trials) introduces the number of parallel simulations or agents to calculate sample results.
    - Serotonin (-s or --serotonin) if present runs SWC with 5-HT+DA, if not is only DA.
    - Plot (-p or --plot) is used for plotting and if only a single trial is chosen
    - Learning rate DA (-d or --dlr).
    - Learning rate 5-HT (-l or --slr).
    - Amplitud STDP DA (-m or --adopa).
    - Amplitud STDP 5-HT (-n or --asero).
    - Figure 3: Activation (-a or --activ) and Inhibition (-i or --inhib) of 5-HT.
    - Time decay of STDP window of 5-HT (-g or --gtime).
    - Time decay of STDP window of DA (-j or --jtime).
    - Elegibility trace decay of 5-HT (-y or --elegib)
    - Change location of the platform at middle episode of a trial (-c --changeposition), boolean option.
    - Random intial locations in each trial (-k or or --randomloc), boolean option.

Run as: python seq_neuromod_MWM.py -o <name_job> -e <episodes> -t <trials> -s

NOTE: Inside the code trials and episodes are exchanged as variables. Run from the interface with episodes (-e)
being the number of sequential runs of an agent and trials (-t) being the number of agents or parallel simulations.
"""

def main():
    parser = OptionParser()
    parser.add_option("-o", "--output", dest="jobID",
                  help="ID of the JOB", metavar="FILE")
    parser.add_option("-e", "--episodes", dest="episodes",
          help="Number of episodes", metavar="int",default=40)
    parser.add_option("-c", "--changeposition", dest="changepos",
          help="Change the postion of reward", default=False, action='store_true')
    parser.add_option("-t", "--trials", dest="trials",
      help="Number of trials", metavar="int",default=1)
    parser.add_option("-s", "--serotonin", dest="serotonin",
      help="Activate serotonergic system", default=False, action='store_true')
    parser.add_option("-p", "--plot", dest="plot",
      help="Plotting", default=False, action='store_true')
    parser.add_option("-d", "--dlr", dest="eta_DA",
                  help="Learning rate for dopamine", metavar="float", default=0.01)
    parser.add_option("-l", "--slr", dest="eta_Sero",
                  help="Learning rate for serotonin", metavar="float", default=0.01)
    parser.add_option("-m", "--adop", dest="A_Dopa",
                  help="STDP magnitude of dopamine", metavar="float", default=1)
    parser.add_option("-n", "--asero", dest="A_Sero",
                  help="STDP magnitude of serotonin", metavar="float", default=1)
    parser.add_option("-a", "--activ", dest="Activ",
                  help="Cyclic serotonin potentiation", default=False, action='store_true')
    parser.add_option("-i", "--inhib", dest="Inhib",
                  help="Cyclic serotonin inhibition", default=False, action='store_true')
    parser.add_option("-g", "--htime", dest="TSero",
                  help="Time constant for the STDP window of serotonin", metavar="float", default=10)
    parser.add_option("-j", "--jtime", dest="TDA",
                  help="Time constant for the STDP window of dopamine", metavar="float", default=10)
    options, args = parser.parse_args()
    jobID = options.jobID
    plot_flag = options.plot

    #NOTE EPISODES AND TRIALS ARE INVERTED

    episodes = int(options.trials) # Number of episodes
    Trials   = int(options.episodes) #Number of trials

    changepos=options.changepos
    Sero=options.serotonin
    #learning rates
    eta_DA=float(options.eta_DA) #learning rate eligibility trace for dopamine
    eta_Sero=float(options.eta_Sero) #learning rate eligibility trace for serotonin
    #magnitudes of neuromodulation
    A_DA=float(options.A_Dopa)
    A_Sero=float(options.A_Sero)
    #activation and inhibition of serotonergic pathway
    Activ=options.Activ
    Inhib=options.Inhib
    #Time constants for STDP window
    tau_DA=options.TDA #time constant pre-post window
    tau_Sero=options.TSero   #time constant post-pre window for serotonin

    if plot_flag:
        import matplotlib.pyplot as plt
        from IPython import display
        plt.close()
    if episodes==1:
        episode_run(jobID,1,plot_flag,Trials,changepos,Sero,eta_DA,eta_Sero,A_DA,A_Sero,tau_DA,tau_Sero,)
    else:
        pool = multiprocessing.Pool(12)
        results=[]
        for episode in range(0,episodes):
            print('Episode',episode)
            results.append(pool.apply_async(episode_run,(jobID,episode,plot_flag,Trials,changepos,Sero,eta_DA,eta_Sero,A_DA,A_Sero,Activ,Inhib,tau_DA,tau_Sero,),error_callback=log_e))
            current_process = psutil.Process()
            children = current_process.children(recursive=True)
            while len(children)>12:
                time.sleep(0.1)
                current_process = psutil.Process()
                children = current_process.children(recursive=True)

        descriptor = {'jobID':jobID,'Trials':Trials,'episodes':episodes,'eta_DA':eta_DA,'eta_Sero':eta_Sero,'A_DA':A_DA,'A_Sero':A_Sero,'Activ':Activ,'Inhib':Inhib,'T_DA':tau_DA,'T_Sero':tau_Sero}
        results_episode = [result.get() for result in results]
        pool.close()
        pool.join()
        with open(jobID+'.pickle', 'wb') as myfile:
            pickle.dump((descriptor,results_episode), myfile)

def log_e(e):
  print(e)

def episode_run(jobID,episode,plot_flag,Trials,changepos,Sero,eta_DA,eta_Sero,A_DA,A_Sero,Activ,Inhib,tau_DA,tau_Sero,):
        #Seed random number of each pool
        np.random.seed()

        #Results to be exported for each episode
        first_reward = None
        rewarding_trials = np.zeros(Trials)
        punishing_trials = np.zeros(Trials)
        quadrant_map = np.empty([Trials,4])
        median_distance = np.zeros(Trials)

        print('Initiated episode:',episode)
        rew1_flag=1  #rewards are in the initial positions
        rew2_flag=0  #reward are switched
        ACh_flag=0 #acetylcholine flag if required for comparisons
        step = 1
        T_max = 15*10**3 #maximum time trial
        starting_position = np.array([0,0]) #starting position
        t_rew = T_max #time of reward - initialized to maximum
        t_extreme = t_rew+300 #time of reward - initialized to maximu
        t_end = T_max
        c = np.array([-1.5,-1.5]) #centre reward 1
        r_goal=0.3 #radius goal area
        c2 = np.array([1.5,1.5]) #centre reward 2
        r_goal2=0.3 #radius goal area2

        ## Place cells positions
        space_pc = 0.4 #place cells separation distance
        bounds_x = np.array([-2,2]) #bounds open field, x axis
        bounds_y = np.array([-2,2]) #bounds open field, y axis
        samples_x = int(np.round((bounds_x[1]-bounds_x[0])/space_pc)+1) #samples of x axis
        samples_y = int(np.round((bounds_y[1]-bounds_y[0])/space_pc)+1) #samples of y axis
        x_pc = np.linspace(bounds_x[0],bounds_x[1],samples_x) #place cells on axis x
        n_x = np.size(x_pc) #nr of place cells on axis x
        y_pc= np.linspace(bounds_y[0],bounds_y[1],samples_y) #place cells on axis y
        n_y = np.size(y_pc) #nr of place cells on axis y
        pos = np.zeros([1,2]) #position of the agent at each timestep

        #create grid
        y = np.matlib.repmat(y_pc, n_x,1).reshape((n_x*n_y,1),order='F')

        x = np.matlib.repmat(x_pc,1,n_y)
        pc = np.concatenate((x.T,y),axis=1)
        N_pc=pc.shape[0] #number of place cells
        rho_pc=400*10**(-3) #maximum firing rate place cells, according to Poisson
        sigma_pc=0.4 #pc separation distance

        # Action neurons - neuron model

        eps0 = 20 #scaling constant epsp
        tau_m = 20 #membrane time constant
        tau_s = 5 #synaptic time rise epsp
        chi = -5 #scaling constant refractory effect
        rho0 = 60*10**(-3) #scaling rate
        theta = 16 #threshold
        delta_u = 2 #escape noise

        ## Action neurons - parameters

        N_action = 40 #number action neurons

        #action selection
        tau_gamma = 50 #raise time convolution action selection
        v_gamma = 20 #decay time convolution action selection
        theta_actor = np.reshape(2*np.pi*np.arange(1,N_action+1)/N_action,(40,1)) #angles actions

        #winner-take-all weights
        psi = 20 #the higher, the more narrow the range of excitation
        w_minus = -300
        w_plus = 100
        diff_theta = np.matlib.repmat(theta_actor.T,N_action,1) - np.matlib.repmat(theta_actor,1, N_action)
        f = np.exp(psi*np.cos(diff_theta)) #lateral connectivity function
        f = f - np.multiply(f,np.eye(N_action))
        normalised = np.sum(f,axis=0)
        w_lateral = (w_minus/N_action+w_plus*f/normalised) #lateral connectivity action neurons

        #actions
        a0=.08
        actions = np.squeeze(a0*np.array([np.sin(theta_actor),np.cos(theta_actor)])) #possible actions (x,y)

        dx = 0.01 #length of bouncing back from walls


        ## synaptic plasticity parameters for dopamine-regulated STDP

        A_pre_post=A_DA   #amplitude pre-post window
        A_post_pre=A_DA   #amplitude post-pre window
        tau_pre_post= tau_DA   #time constant pre-post window
        tau_post_pre= tau_DA   #time constant post-pre window
        tau_e= 2*10**3 #time constant eligibility trace

        ## synaptic plasticity parameters for serotonin-regulated STDP

        A_pre_post_sero=A_Sero   #amplitude pre-post window for serotonin
        A_post_pre_sero=0   #amplitude post-pre window for serotonin
        tau_pre_post_sero= tau_Sero   #time constant pre-post window for serotonin
        tau_post_pre_sero= tau_Sero   #time constant post-pre window for serotonin
        tau_e_sero= 5*10**3 #time constant eligibility trace for serotonin


        #ACh learning rate if active
        eta_ACh = 10**-3*2 #learning rate acetylcholine

        #feed-forward weights
        w_max=3 #upper bound feed-forward weights
        w_min=1 #.pwer bound feed-forward weights
        w_in = np.ones([N_pc, N_action]).T*2 #initialization feed-forward weights

        trace_pre_post= np.zeros([N_action, N_pc]) #initialize pre-post trace
        trace_post_pre= np.zeros([N_action,N_pc])#initialize post-pre trace
        trace_tot = np.zeros([N_action,N_pc]) #sum of the traces
        eligibility_trace = np.zeros([N_action, N_pc]) #total convolution

        trace_pre_post_sero= np.zeros([N_action, N_pc]) #initialize pre-post trace
        trace_post_pre_sero= np.zeros([N_action,N_pc])#initialize post-pre trace
        trace_tot_sero = np.zeros([N_action,N_pc]) #sum of the traces
        eligibility_trace_sero = np.zeros([N_action, N_pc]) #total convolution

        ## initialise variables

        i=int(0) #counter ms
        tr=0 #counter trial

        w_tot = np.concatenate((np.ones([N_pc,N_action]).T*w_in,w_lateral),axis=1)#total weigths


        X = np.zeros([N_pc,1]) #matrix of spikes place cells
        X_cut = np.zeros([N_pc+N_action, N_action])  #matrix of spikes place cells
        Y_action_neurons= np.zeros([N_action, 1])  #matrix of spikes action neurons

        time_reward= np.zeros([Trials,1]) #stores time of reward 1
        time_reward2 = np.copy(time_reward) #stores time of reward 2 (moved)
        time_reward_old= np.copy(time_reward) #stores time when agent enters the previously rewarded location

        epsp_rise=np.zeros([N_action+N_pc,N_action]) #epsp rise compontent convolution
        epsp_decay=np.zeros([N_action+N_pc,N_action]) #epsp decay compontent convolution
        epsp_tot=np.zeros([N_action+N_pc, N_action]) #epsp

        rho_action_neurons= np.zeros([N_action,1]) #firing rate action neurons
        rho_rise= np.copy(rho_action_neurons)  #firing rate action neurons, rise compontent convolution
        rho_decay = np.copy(rho_action_neurons) #firing rate action neurons, decay compontent convolution

        Canc = np.ones([N_pc+N_action,N_action]).T
        last_spike_post=np.zeros([N_action,1])-1000 #vector time last spike postsynaptic neuron

        store_pos = np.zeros([T_max*Trials,2]) #stores trajectories (for plotting)
        firing_rate_store = np.zeros([N_action,T_max*Trials]) #stores firing rates action neurons (for plotting)

        ## initialize plot open field
        if plot_flag:
            plt.close()
            fig, ((ax1, ax2), (ax3, ax4))= plt.subplots(figsize=(8, 8), ncols=2, nrows=2)
            fig.subplots_adjust(hspace = 0.5)
            fig.show()
            plt.ion()
            #Plot of reward places and intial position
            reward_plot = ax1.plot(c[0]+r_goal*np.cos(np.linspace(-np.pi,np.pi,100)), c[1]+r_goal*np.sin(np.linspace(-np.pi,np.pi,100)),'b') #plot reward 1
            point_plot,= ax1.plot(starting_position[0],starting_position[1], 'r',marker='o',markersize=5) #plot initial starting point

            #plot walls
            ax1.plot([bounds_x[0],bounds_x[1]], [bounds_y[1],bounds_y[1]], 'k')
            ax1.plot([bounds_x[0],bounds_x[1]], [bounds_y[0],bounds_y[0]], 'k')
            ax1.plot([bounds_x[0],bounds_x[0]], [bounds_y[0],bounds_y[1]], 'k')
            ax1.plot([bounds_x[1],bounds_x[1]], [bounds_y[0],bounds_y[1]], 'k')


        ## delete actions that lead out of the maze

        #find index place cells that lie on the walls
        sides = np.empty((4,np.max([n_x,n_y])))
        sides[0,:] = np.where(pc[:,1] == -2)[0].T #bottom wall, y=-2
        sides[1,:] = np.where(pc[:,1] == 2)[0].T #top wall, y=+2
        sides[2,:] = np.where(pc[:,0] == 2)[0].T #left wall, x=-2
        sides[3,:] = np.where(pc[:,0] == -2)[0].T #right wall, x=+2

        #store index of actions forbidden from each side
        forbidden_actions = np.empty((4,19))
        forbidden_actions[0,:] = np.arange(11,30) #actions that point south - theta in (180, 360) degrees approx
        forbidden_actions[1,:] = np.concatenate([np.arange(1,10), np.arange(31,41)]) #actions that point north - theta in (0,180) degrees approx
        forbidden_actions[2,:] = np.arange(1,20) #actions that point east - theta in (-90, 90) degrees approx
        forbidden_actions[3,:] = np.arange(21,40) #actions that point west - theta in (90, 270) degrees approx
        #kill connections between place cells on the walls and forbidden actions
        w_walls = np.ones([N_action, N_pc+N_action])
        for g in range(4):
            idx = list(itertools.product(forbidden_actions[g,:].astype(int).tolist(),sides[g,:].astype(int).tolist()))
            w_walls[np.array(idx)[:,0]-1,np.array(idx)[:,1]] = 0
        # optogenetic ranges
        ranges = [(T_max*Trials/6,2*T_max*Trials/6), (3*T_max*Trials/6,4*T_max*Trials/6), (5*T_max*Trials/6,T_max*Trials)]
        ## start simulation
        w_tot_old = w_tot[0:N_action,0:N_pc] #store weights before start
        while i<T_max*Trials:
            i+=int(1)
            t=np.mod(i,T_max)
            ## reset new trial
            if t==1:
                quadrant = np.zeros([2,2])
                median_tr = []
                pos = starting_position #initialize position at origin (centre open field)
                rew_found=0 #flag that signals when the reward is found
                tr+=int(1) #trial number
                print('Episode:',episode,'Trial:',tr)
                t_rew=T_max #time of reward - initialized at T_max at the beginning of the trial
                #initialisation variables - reset between trials
                Y_action_neurons= np.zeros([N_action, 1])
                X_cut = np.zeros([N_pc+N_action, N_action])
                epsp_rise=np.zeros([N_action+N_pc,N_action])
                epsp_decay=np.zeros([N_action+N_pc,N_action])
                epsp_tot=np.zeros([N_action+N_pc, N_action])
                rho_action_neurons= np.zeros([N_action,1])
                rho_rise=  np.zeros([N_action,1])
                rho_decay =  np.zeros([N_action,1])
                Canc = np.ones([N_pc+N_action,N_action]).T
                last_spike_post=np.zeros([N_action,1])-1000
                trace_pre_pos= np.zeros([N_action, N_pc])
                trace_post_pre= np.zeros([N_action,N_pc])
                trace_tot = np.zeros([N_action,N_pc])
                eligibility_trace = np.zeros([N_action, N_pc])

                #change reward location in the second half of the experiment
                if (tr==(Trials/2)+1) and changepos:
                    rew1_flag=0
                    rew2_flag=1
                    np.linspace(-np.pi, np.pi, 100)
                    if plot_flag:
                        reward_plot.pop(0).remove()
                        punish_plot.pop(0).remove()
                        reward_plot = ax1.plot(c2[0]+r_goal*np.cos(np.linspace(-np.pi,np.pi,100)), c2[1]+r_goal*np.sin(np.linspace(-np.pi,np.pi,100)),'b') #plot negative reward 2
            ## place cells
            rhos = np.multiply(rho_pc,np.exp(-np.sum((np.matlib.repmat(pos,n_x*n_y,1)-pc)**2,axis=1)/(sigma_pc**2))) #rate inhomogeneous poisson process
            prob = rhos
            #turn place cells off after reward is reached
            if t>t_rew:
                prob=np.zeros_like(rhos)
            X = (np.random.rand(1,N_pc)<=prob.T).T #spike train pcs
            store_pos[i-1,:] = pos #store position (for plotting)

            #save quadrant
            if not (pos==[0,0]).all():
                quadrant[int(not(bisect([0],pos[1]))),bisect([0],pos[0])]+=1

            # save median distance to centre
            median_tr.append(np.linalg.norm(pos)/2)

            ## reward
            # agent enters reward 1 in the first half of the trial
            if np.sum((pos-c)**2)<=r_goal**2 and rew_found==0 and rew1_flag==1:
                rew_found=1 #flag reward found (so that trial is ended soon)
                t_rew=t #time of reward
                time_reward[tr-1] = t #store time of reward
                rewarding_trials[tr-1:]+=1
                if not(first_reward):
                    first_reward=tr
                    print('First reward,episode',episode,'trial',first_reward)


            #cases for location switching

            # agent enters reward 2 in the second half of the trial
            if np.sum((pos-c2)**2)<=r_goal2**2 and rew_found==0 and rew2_flag==1:
                rew_found=1  #flag reward 2 found (so that trial is ended soon)
                t_rew=t #time of reward 2
                time_reward2[tr-1] = t #store time of reward 2
                rewarding_trials[tr-1:]+=1

            if np.sum((pos-c)**2)<=r_goal**2 and rew1_flag==0 and rew2_flag==1:
                 #this location is no longer rewarded, so the trial is not ended
                time_reward_old[tr-1]=t #store time of entrance old reward location

            ## action neurons
            # reset after last post-synaptic spike
            X_cut = np.matlib.repmat(np.concatenate((X,Y_action_neurons)),1,N_action)
            X_cut = np.multiply(X_cut,Canc.T)
            epsp_rise=np.multiply(epsp_rise,Canc.T)
            epsp_decay=np.multiply(epsp_decay,Canc.T)
            # neuron model
            epsp_tot, epsp_decay, epsp_rise = convolution (epsp_decay, epsp_rise, tau_m, tau_s, eps0, X_cut, np.multiply(w_tot,w_walls)) #EPSP in the model * weights
            Y_action_neurons,last_spike_post, Canc, _ = neuron(epsp_tot, chi, last_spike_post, tau_m, rho0, theta, delta_u, i) #sums EPSP, calculates potential and spikes

            # smooth firing rate of the action neurons
            rho_action_neurons, rho_decay, rho_rise = convolution (rho_decay, rho_rise, tau_gamma, v_gamma, 1, Y_action_neurons)
            firing_rate_store[:,i-1] = np.squeeze(rho_action_neurons) #store action neurons' firing rates
            # select action
            a = (np.dot(rho_action_neurons.T,np.squeeze(actions).T)/N_action)
            a[np.isnan(a)]=0
            ## synaptic plasticity

            #STDP with symmetric window
            trace_pre_pos, trace_post_pre,eligibility_trace, trace_tot, W = weights_update_stdp(A_pre_post, A_post_pre, tau_pre_post, tau_post_pre, np.matlib.repmat(X.T,N_action,1) , np.matlib.repmat(Y_action_neurons,1, N_pc), trace_pre_pos, trace_post_pre, trace_tot, tau_e)

            #STDP with unsymmetric window and depression due to serotonin
            if not(Inhib) and not(Activ):
                trace_pre_post_sero, trace_post_pre_sero,eligibility_trace_sero, trace_tot_sero, W_sero = weights_update_stdp(A_pre_post_sero, A_post_pre_sero, tau_pre_post_sero, tau_post_pre_sero, np.matlib.repmat(X.T,N_action,1) , np.matlib.repmat(Y_action_neurons,1, N_pc), trace_pre_post_sero, trace_post_pre_sero, trace_tot_sero, tau_e_sero)
            elif Activ and any(lower <= i<= upper for (lower, upper) in ranges):
                #If there is overpotentiation of serotonin, assumed as doubled
                trace_pre_post_sero, trace_post_pre_sero,eligibility_trace_sero, trace_tot_sero, W_sero = weights_update_stdp(1.15*A_pre_post_sero, 1.15*A_post_pre_sero, tau_pre_post_sero, tau_post_pre_sero, np.matlib.repmat(X.T,N_action,1) , np.matlib.repmat(Y_action_neurons,1, N_pc), trace_pre_post_sero, trace_post_pre_sero, trace_tot_sero, tau_e_sero)
            elif Inhib and any(lower <= i<= upper for (lower, upper) in ranges):
                #If there is inhibition of serotonin, no eligibility trace is produced
                pass

            # online weights update (effective only with acetylcholine - ACh_flag=1)
            w_tot[0:N_action,0:N_pc]= w_tot[0:N_action,0:N_pc]-eta_ACh*W*(ACh_flag)

            #weights limited between lower and upper bounds
            w_tot[np.where(w_tot[:,0:N_pc]>w_max)]=w_max
            w_tot[np.where(w_tot[:,0:N_pc]<w_min)]=w_min
            ## position update
            pos = np.squeeze(pos+a)
            #check if agent is out of boundaries. If it is, bounce back in the opposite direction
            if pos[0]<=bounds_x[0]:
                pos = pos+dx*np.array([1,0])
            else:
                if pos[0]>= bounds_x[1]:
                    pos = pos+dx*np.array([-1,0])
                else:
                    if pos[1]<=bounds_y[0]:
                        pos = pos+dx*np.array([0,1])
                    else:
                        if pos[1]>=bounds_y[1]:
                            pos = pos+dx*np.array([0,-1])
            #time when trial end is 300ms after reward is found
            t_extreme = t_rew+300
            if t> t_extreme and t<T_max:
                i = int((np.ceil(i/T_max))*T_max)-1 #set i counter to the end of the trial
                t_end = t_extreme #for plotting
            if t==0:
                t=T_max
                ## update weights - end of trial

                # if the reward is not found, no change.
                # change due to serotonin or dopamine
                if Sero:
                    w_tot[0:N_action,0:N_pc]= (w_tot_old+eta_DA*eligibility_trace)*rew_found + (w_tot_old-eta_Sero*eligibility_trace_sero)*(1-rew_found)*(not Inhib)
                else:
                    #change due to dopamine or sustained weights at the end of the trial
                    w_tot[0:N_action,0:N_pc]=w_tot[0:N_action,0:N_pc]*(1-rew_found)+(w_tot_old+eta_DA*eligibility_trace)*rew_found

                #weights limited between lower and upper bounds
                w_tot[np.where(w_tot[:,0:N_pc]>w_max)]=w_max
                w_tot[np.where(w_tot[:,0:N_pc]<w_min)]=w_min

                #store weights before the beginning of next trial (for updates in case reward is found)
                w_tot_old = np.copy(w_tot[0:N_action,0:N_pc])

                #calculate policy
                ac =np.dot(np.squeeze(actions),(np.multiply(w_tot_old,w_walls[:,0:N_pc]))/a0) #vector of preferred actions according to the weights
                ac[:,np.unique(np.sort(np.reshape(sides, (np.max(sides.shape)*4, 1),order='F'))).astype(int).tolist()]=0 #do not count actions AT the boundaries (just for plotting)

                ## save quadrant
                quadrant_map[tr-1,:]=quadrant.flatten()/np.sum(quadrant)

                ##save median_distance
                median_distance[tr-1]=np.median(median_tr)
                ## plot

                if plot_flag:
                    #display trajectory of the agent in each trial
                    f3 =ax1.plot(store_pos[int((np.floor((i-1)/T_max))*T_max+1):int((np.floor((i-1)/T_max))*T_max+t_end+1),0], store_pos[int((np.floor((i-1)/T_max))*T_max+1):int((np.floor((i-1)/T_max))*T_max+t_end+1),1]) #trajectory
                    point_plot, = ax1.plot(starting_position[0],starting_position[1],'r',marker='o',markersize=5) #starting point
                    ax1.set_title('Trial '+str(tr))

                    #display action neurons firing rates (activity bump)
                    pos = ax2.imshow(firing_rate_store[:,int((np.floor((i-1)/T_max))*T_max):int((np.floor((i-1)/T_max))*T_max+t_end)],cmap='Blues', interpolation='none',aspect='auto')
                    #colorbar
                    ax2.set_title('Action neurons firing rates')
                    if tr==1:
                        fig.colorbar(pos, ax=ax2)
                    #display weights over the open field, averaged over action neurons
                    w_plot = np.mean(w_tot[:,0:N_pc],axis=0) #use weights as they were at the beginning of the trial
                    w_plot = np.reshape(w_plot,(int(np.sqrt(N_pc)),int(np.sqrt(N_pc))))
                    pos2 = ax3.imshow(w_plot,cmap='Reds_r',origin='lower', interpolation='none',aspect='auto')
                    #set(gca,'YDir','normal')
                    if tr==1:
                        fig.colorbar(pos2, ax=ax3)
                    ax3.set_title('Mean weights')
                    #plot policy as a vector field
                    #filter zero values
                    ac_norm=np.max(np.linalg.norm(ac,axis=0))
                    f4=ax4.quiver(pc[:,0], pc[:,1], ac[0,:].T/ac_norm, ac[1,:].T/ac_norm)
                    ax4.set_xlim([-2,2])
                    ax4.set_ylim([-2,2])
                    ax4.set_title('Agent''s policy')
                    #time.sleep(1.0)
                    fig.canvas.draw()
                    plt.pause(0.00001)
                    f3.pop(0).remove()
                    f4.remove()
                    pos2.remove()
                    t_end = T_max;

        x_pos=store_pos[np.where(store_pos.any(axis=1))[0],0]
        y_pos=store_pos[np.where(store_pos.any(axis=1))[0],1]
        try:
            pos_hist=np.histogram2d(x_pos, y_pos, bins=(np.linspace(-2,2,50),np.linspace(-2,2,50)))[0]
        except:
            print(pos_hist.shape)
        return episode,first_reward,rewarding_trials,punishing_trials,quadrant_map,median_distance,time_reward,time_reward2,time_reward_old,pos_hist
if __name__ == '__main__':
    main()
