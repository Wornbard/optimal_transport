import numpy as np

def mina_u(H,a,epsilon): return -epsilon*np.log( np.sum(a * np.exp(-H/epsilon),0) )
def minb_u(H,b,epsilon): return -epsilon*np.log( np.sum(b * np.exp(-H/epsilon),1) )


def mina(H,a,epsilon):
    #print(mina_u(H-np.min(H,0),a,epsilon) + np.min(H,0))
    return mina_u(H-np.min(H,0),a,epsilon) + np.min(H,0)
def minb(H,b,epsilon): return minb_u(H-np.min(H,1)[:,None],b,epsilon) + np.min(H,1)

def Sinkhorn(C,a,b,epsilon,f,niter = 500):#cost matrix, histograms    
    Err = np.zeros(niter)
    for it in range(niter):
        g = mina(C-f[:,None],a,epsilon)
        f = minb(C-g[None,:],b,epsilon)
        # generate the coupling
        P = a * np.exp((f[:,None]+g[None,:]-C)/epsilon) * b
        # check conservation of mass
        Err[it] = np.linalg.norm(np.sum(P,0)-b,1)
    return (P,Err)

def L2_cost(source,sink):
    C=np.empty((len(source),len(sink)))
    for i,pos1 in enumerate(source):
        for j, pos2 in enumerate(sink):
            C[i,j]=np.sqrt((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2)
    return C