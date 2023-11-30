import jax
import jax.numpy as xp
import matplotlib.pyplot as plt 

def f(x):
    '''
    FUNCTION TO BE OPTIMISED
    '''
    d = len(x)
    return sum(100*(x[i+1]-x[i]**2)**2 + (x[i]-1)**2 for i in range(d-1))

def grad(f,x, h=1e-3): 
    '''
    CENTRAL FINITE DIFFERENCE CALCULATION
    '''
    # h = xp.cbrt(xp.finfo(float).eps)
    d = len(x)
    nabla = xp.zeros(d)
    for i in range(d): 
        x_for = x.at[i].set(x[i]+h) 
        x_back = x.at[i].set(x[i]-h) 
        f_for = f(x_for)
        f_back = f(x_back)
        f_dif = (f_for- f_back)/(2*h)
        nabla = nabla.at[i].set(f_dif)
        print("Dimension {} -- Finite dif: {}".format(i+1, f_dif))
    return nabla 

def line_search(f,x,p,nabla):
    '''
    BACKTRACK LINE SEARCH WITH WOLFE CONDITIONS
    '''
    a = 1
    c1 = 1e-4 
    c2 = 0.9 
    fx = f(x)
    x_new = x + a * p 
    nabla_new = grad(f,x_new)
    while f(x_new) >= fx + (c1*a*nabla.T@p) or nabla_new.T@p <= c2*nabla.T@p : 
        a *= 0.5
        x_new = x + a * p 
        nabla_new = grad(f,x_new)
    return a


def BFGS(f,x0,max_it,plot=False):
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
    d = len(x0) # dimension of problem 
    nabla = grad(f,x0) # initial gradient 
    H = xp.eye(d) # initial hessian
    x = x0[:]
    it = 2 
    # if plot == True: 
    #     if d == 2: 
    #         x_store =  xp.zeros((1,2)) # storing x values 
    #         x_store = x_store.at[0,:].set(x) 
    #     else: 
    #         print('Too many dimensions to produce trajectory plot!')
    #         plot = False
    # #
    while xp.linalg.norm(nabla) > 1e-5: # while gradient is positive
        print("BFGS Iteration {}".format(it))
        if it > max_it: 
            print('Maximum iterations reached!')
            break
        it += 1
        p = -H@nabla # search direction (Newton Method)
        a = line_search(f,x,p,nabla) # line search 
        s = a * p 
        x_new = x + a * p 
        nabla_new = grad(f,x_new)
        y = nabla_new - nabla 
        y = xp.array([y])
        s = xp.array([s])
        y = xp.reshape(y,(d,1))
        s = xp.reshape(s,(d,1))
        r = 1/(y.T@s)
        li = (xp.eye(d)-(r*((s@(y.T)))))
        ri = (xp.eye(d)-(r*((y@(s.T)))))
        hess_inter = li@H@ri
        H = hess_inter + (r*((s@(s.T)))) # BFGS Update
        nabla = nabla_new[:] 
        x = x_new[:]
    #     if plot == True:
    #         x_store = xp.append(x_store,[x],axis=0) # storing x
    # if plot == True:
    #     x1 = xp.linspace(min(x_store[:,0]-0.5),max(x_store[:,0]+0.5),30)
    #     x2 = xp.linspace(min(x_store[:,1]-0.5),max(x_store[:,1]+0.5),30)
    #     X1,X2 = xp.meshgrid(x1,x2)
    #     Z = f([X1,X2])
    #     plt.figure()
    #     plt.title('OPTIMAL AT: '+str(x_store[-1,:])+'\n IN '+str(len(x_store))+' ITERATIONS')
    #     plt.contourf(X1,X2,Z,30,cmap='jet')
    #     plt.colorbar()
    #     plt.plot(x_store[:,0],x_store[:,1],c='w')
    #     plt.xlabel('$x_1$'); plt.ylabel('$x_2$')
    #     plt.show()
    return x


'''
from functools import partial

_f_intra_part = partial(_f_intra, m_est = m_est, \
                        C = C, res = res, \
                        U_shot = U_shot, \
                        R_pad = R_pad, \
                        s_corrupted = s_corrupted)

f_val = _f_intra_part(Mtraj_est_n)
g_val = grad(_f_intra_part, Mtraj_est_n)

from time import time
t1 = time()
f_for = f(x_for)
t2 = time()
print("Elapsed time: {} sec".format(t2 - t1))
## NB. takes 4.4 sec per f evaluation
## So takes 12.8 minutes to eval grad per shot


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

x_opt = BFGS(_f_intra_part,Mtraj_est_n,1)

'''


