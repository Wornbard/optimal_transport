# for details see : https://nbviewer.jupyter.org/github/gpeyre/numerical-tours/blob/master/python/optimaltransp_6_entropic_adv.ipynb

import numpy as np

normalise = lambda x : x/np.sum(np.abs(x))

def mina_u(H,a,epsilon): 
    tmp=np.sum(a * np.exp(-H/epsilon),0)
    tmp=np.where(np.abs(tmp)<10e-15, 10e-15, tmp)
    return -epsilon*np.log( tmp )
def minb_u(H,b,epsilon):
    tmp=np.sum(b * np.exp(-H/epsilon),1)
    tmp=np.where(np.abs(tmp)<10e-15, 10e-15, tmp)
    return -epsilon*np.log( tmp )


def mina(H,a,epsilon):
    #print(mina_u(H-np.min(H,0),a,epsilon) + np.min(H,0))
    return mina_u(H-np.min(H,0),a,epsilon) + np.min(H,0)
def minb(H,b,epsilon): return minb_u(H-np.min(H,1)[:,None],b,epsilon) + np.min(H,1)

def Sinkhorn(C,a,b,epsilon,f,niter = 500):#cost matrix, histograms, regularization strength, initial f, number of iterations    
    Err = np.zeros(niter)
    for it in range(niter):
        g = mina(C-f[:,None],a,epsilon)
        f = minb(C-g[None,:],b,epsilon)
        # generate the coupling
        P = a * np.exp((f[:,None]+g[None,:]-C)/epsilon) * b
        # check conservation of mass
        Err[it] = np.linalg.norm(np.sum(P,0)-b,1)
    return (P,Err)

def unb_Sinkhorn(C,a,b,epsilon,rho,f,niter=500):#same as above and rho is the weight of the divergence terms in unbalanced OT
    kappa = rho/(rho+epsilon)
    for it in range(niter):
        g = kappa*mina(C-f[:,None],a,epsilon)
        f = kappa*minb(C-g[None,:],b,epsilon)
    P = a * np.exp((f[:,None]+g[None,:]-C)/epsilon) * b
    return P

#Not relevant anymore
def L2_cost(source,sink):
    C=np.empty((len(source),len(sink)))
    for i,pos1 in enumerate(source):
        for j, pos2 in enumerate(sink):
            C[i,j]=np.sqrt((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2)
    return C

#Not relevant either
def sink_L2_cost(sp,ep,cr_cost=1,del_cost=1,INF=1e7):
    sub_cost=L2_cost(sp,ep)
    left_col=np.hstack((np.ones((len(sp),1)) * INF ,del_cost*np.ones((len(sp),1))))
    top_row=np.vstack((cr_cost*np.ones((1,len(ep))),INF*np.ones((1,len(ep)))))
    corner=np.array([[0,0],[INF,0]])
    top=np.hstack((corner,top_row))
    bottom=np.hstack((left_col,sub_cost))
    end_cost=np.vstack((top,bottom))

    return end_cost/np.average(end_cost)
