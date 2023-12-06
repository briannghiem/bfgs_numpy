import jax
import jax.numpy as xp
import matplotlib.pyplot as plt 
from time import time

def line_search(f,x,p,g):
    '''
    BACKTRACK LINE SEARCH WITH WOLFE CONDITIONS
    '''
    a = 1
    c1 = 1e-4 
    c2 = 0.9 
    fx = f(x)
    x_new = x + a * p 
    g_new = grad(f,x_new,args)
    while f(x_new) >= fx + (c1*a*g.T@p) or g_new.T@p <= c2*g.T@p : 
        a *= 0.5
        x_new = x + a * p 
        g_new = grad(f,x_new,args)
    return a

def BFGS(f,x0,args,max_it,spath):
    '''
    DESCRIPTION
    BFGS Quasi-Newton Method, implemented as described in Nocedal:
    Numerical Optimisation.
    #
    #
    INPUTS:
    f:      function to be optimised 
    x0:     intial guess
    max_it: maximum iterations 
    plot:   if the problem is 2 dimensional, returns 
            a trajectory plot of the optimisation scheme.
    #
    OUTPUTS: 
    x:      the optimal solution of the function f 
    #
    '''
    #Initial values
    d = len(x0) # dimension of problem 
    g = grad(f,x0,args) # initial gradient 
    H = xp.eye(d) # initial hessian
    x = x0[:]
    #Set up stores
    x_store = []
    f_store = []
    g_store = []
    #Run BFGS algorithm
    it = 1 
    while xp.linalg.norm(g) > 1e-5: # while gradient is positive
        print("BFGS Iteration {}".format(it), end = '\n')
        if it > max_it: 
            print('Maximum iterations reached!')
            break
        t1 = time()
        p = -H@g # search direction (Newton Method)
        # a = line_search(f,x,p,g) # line search 
        a = 1e-4 #fixing step size
        s = a * p 
        x_new = x + s
        g_new = grad(f,x_new,args)
        y = g_new - g 
        y = xp.array([y])
        s = xp.array([s])
        y = xp.reshape(y,(d,1))
        s = xp.reshape(s,(d,1))
        r = 1/(y.T@s)
        li = (xp.eye(d)-(r*((s@(y.T)))))
        ri = (xp.eye(d)-(r*((y@(s.T)))))
        hess_inter = li@H@ri
        H = hess_inter + (r*((s@(s.T)))) # BFGS Update
        g = g_new[:] 
        x = x_new[:]
        it += 1
        #Save to store
        x_store.append(x)
        f_store.append(f(x,args[0],args[1],args[2],args[3],args[4],args[5]))
        g_store.append(g)
        xp.save(spath + r'/opt_out.npy', [x_store, f_store, g_store])
        t2 = time()
        print("Elapsed time for BFGS Iter {}: {} sec".format(it, t2 - t1))
    return x_store, f_store, g_store


'''

from functools import partial

def _f_intra(Mtraj_est_n, m_est=None, C=None, res=None, U_shot=None, R_pad=None, s_corrupted=None):
    #Data consistency for a given shot
    n_shots = len(U_shot)
    s_n = eop.Encode(m_est, C, U_shot, Mtraj_est_n.reshape(n_shots, 6), res, R_pad)
    #
    U_shot_full = xp.zeros(s_corrupted.shape)
    for i in range(len(U_shot)):
        U_shot_full += eop._gen_U_n(U_shot[i], m_est.shape)
    #
    DC = s_n.flatten() - (U_shot_full*s_corrupted).flatten()
    return xp.abs(xp.dot(xp.conjugate(DC), DC)) #L2-norm

def grad(f_init,x,args,h=1e-3):     
    #CENTRAL FINITE DIFFERENCE CALCULATION
    # h = xp.cbrt(xp.finfo(float).eps)
    d = len(x)
    g = xp.zeros(d)
    #
    U_RO = args[3][0][0]
    U_PE1 = [int(args[3][i][1]) for i in range(len(args[3]))]
    U_PE2 = args[3][0][2]
    #
    for i in range(d): 
        #Set-up partial loss func
        TR_ind = int(xp.floor(i/6)) #Index of TR, given 6 DOFs per TR
        U_shot_i = args[3][TR_ind]
        U_PE1_rest = xp.asarray(U_PE1[:TR_ind] + U_PE1[TR_ind+1:])
        U_shot_i_rest = [U_RO, U_PE1_rest, U_PE2]
        U_shot_temp = [U_shot_i, U_shot_i_rest]
        #
        f = partial(f_init, m_est = args[0], \
                    C = args[1], res = args[2], \
                    U_shot = U_shot_temp, \
                    R_pad = args[4], \
                    s_corrupted = args[5])
        #
        #Evaluate finite difference 
        x_init = x[TR_ind*6:(TR_ind+1)*6]
        x_for_init = x.at[i].set(x[i]+h)[TR_ind*6:(TR_ind+1)*6]
        x_back_init = x.at[i].set(x[i]-h)[TR_ind*6:(TR_ind+1)*6]
        x_for = xp.append(x_for_init, x_init)
        x_back = xp.append(x_back_init, x_init)
        #
        f_for = f(x_for)
        f_back = f(x_back)
        f_dif = (f_for- f_back)/(2*h)
        g = g.at[i].set(f_dif)
        print("Dimension {} -- Finite dif: {}".format(i+1, f_dif), end='\r')
    return g 


#--------------------------------
#Picking up algorithm
m_est = np.load(spath + r'/m_intmd.npy')
m_loss_store = list(np.load(spath + r'/m_loss_store.npy', allow_pickle=1))
Mtraj_store = list(np.load(spath + r'/Mtraj_store.npy', allow_pickle=1))
Mtraj_est = Mtraj_store[-1][0]

#--------------------------------
#Setting up shot pattern 
nTRs = 16
nPE = m_est.shape[1]
U_alt = U[0][1]
for i in range(1,len(U)):
    U_alt = xp.concatenate((U_alt, U[i][1]))
#

nstart = 0
U_full = []
U_full.append([U[0][0], U_alt[nstart], U[0][2]])
for i in range(nstart+1, nPE):
    U_full.append([U[0][0], U_alt[i], U[0][2]])
#

#Working with first 16 TRs (ie. first shot)
shot_ind = 1
shot_TRs = 16
U_shot = U_full[shot_ind*shot_TRs:(shot_ind+1)*shot_TRs]
Mtraj_est_init = Mtraj_est[shot_ind,:]
Mtraj_est_n = xp.tile(Mtraj_est_init, (shot_TRs,1)).flatten()

#--------------------------------
#Picking up algorithm
m_est = np.load(spath + r'/m_intmd.npy')
m_loss_store = list(np.load(spath + r'/m_loss_store.npy', allow_pickle=1))
Mtraj_store = list(np.load(spath + r'/Mtraj_store.npy', allow_pickle=1))
Mtraj_est = Mtraj_store[-1][0]
shot_ind = 1
shot_TRs = 16
U_shot = U_full[shot_ind*shot_TRs:(shot_ind+1)*shot_TRs]
Mtraj_est_init = Mtraj_est[shot_ind,:]
Mtraj_est_n = xp.tile(Mtraj_est_init, (shot_TRs,1)).flatten()


maxiter = 2
args = [m_est, C, res, U_shot, R_pad, s_corrupted]


from time import time
t1 = time()
g = grad(_f_intra, Mtraj_est_n, args)
t2 = time()
print("Elapsed time for gradient evaluation: {} sec".format(t2 - t1))


t1 = time()
opt_out = BFGS(_f_intra, Mtraj_est_n, args, maxiter, spath)
t2 = time()
print("Elapsed time for BFGS evaluation: {} sec".format(t2 - t1))
#

Mtraj_est_n_new = opt_out[0][0]

Mtraj_est_n_plt = Mtraj_est_n.reshape(16,6)
Mtraj_est_n_new_plt = Mtraj_est_n_new.reshape(16,6)


plt.style.use('dark_background')

plt.figure()
plt.plot(Mtraj_est_n_new_plt[:,0], '#1F77B4', label = "SI - UNet+JE")
plt.plot(Mtraj_est_n_new_plt[:,1], '#2CA02C', label = "AP - UNet+JE")
plt.plot(Mtraj_est_n_new_plt[:,2], '#E377C2', label = "LR - UNet+JE")
plt.plot(Mtraj_est_n_plt[:,0], '#1F77B4', linestyle = '--', label = "SI - JE", alpha = 0.90)
plt.plot(Mtraj_est_n_plt[:,1], '#2CA02C', linestyle = '--', label = "AP - JE", alpha = 0.90)
plt.plot(Mtraj_est_n_plt[:,2], '#E377C2', linestyle = '--', label = "LR - JE", alpha = 0.90)
# plt.legend(loc = "lower left")
plt.xlabel("k-space segment")
plt.ylabel("Translation (mm)")
# plt.title("Estimated Translation Parameters")
plt.title("Translation Parameters")
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.show()

plt.figure()
plt.plot(Mtraj_est_n_new_plt[:,3], '#1F77B4', label = "SI - UNet+JE")
plt.plot(Mtraj_est_n_new_plt[:,4], '#2CA02C', label = "AP - UNet+JE")
plt.plot(Mtraj_est_n_new_plt[:,5], '#E377C2', label = "LR - UNet+JE")
plt.plot(Mtraj_est_n_plt[:,3], '#1F77B4', linestyle = '--', label = "SI - JE", alpha = 0.90)
plt.plot(Mtraj_est_n_plt[:,4], '#2CA02C', linestyle = '--', label = "AP - JE", alpha = 0.90)
plt.plot(Mtraj_est_n_plt[:,5], '#E377C2', linestyle = '--', label = "LR - JE", alpha = 0.90)
# plt.legend(loc = "lower left")
plt.xlabel("k-space segment")
plt.ylabel("Rotation (deg)")
# plt.title("Estimated Rotation Parameters")
plt.title("Rotation Parameters")
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.show()

'''
