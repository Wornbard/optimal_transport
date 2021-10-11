import numpy as np
import ot
#import ot.plot
from ot_utils import *
#from extract_pos import *
import pandas as pd
from tqdm import tqdm
'''
User input : 
 - filename : name of the csv with input
 - dt_max : maximal frame separation between things we want to connect
 - epsilon : coefficient of the entropic regularization term
 - alpha : coefficient of the measure divergence term
 - log_sinkhorn : whether or not to use the logarithmic implementation of Sinkhorn's algorithm
 - n_steps : number of iterations for every execution of Sinkhorn's algorithm
 - min_weight : minimum non-neglebible weight of a conection
 - output_name
'''

#filename
filename = input( 'Name of the csv with data (with the ".csv"extension). Note it must contain columns named t,x,y,intensity: ' )
#TODO : write verification of whether these columns are in the dataframe
try:
    df = pd.read_csv(filename)
except:
    raise ValueError('Incorrect file name')


df.t=df.t.subtract(df.t.min())
split_frames= [d for _,d in df.groupby(df.t)]
int_ranges=[[d.intensity.min(),d.intensity.max()]for d in split_frames]

for i,row in df.iterrows():
    df.loc[i,'intensity']=(row.intensity)/(int_ranges[int(row.t)][1])
    #pass

frames=[df.loc[df.t==i].to_dict('records') for i in df.t.unique()]

positions=[np.array([[obs['x'],obs['y']]for obs in observations])for t,observations in enumerate(frames)]
intensities=[np.array([obs['intensity'] for obs in observations]) for t,observations in enumerate(frames)]

t_max=len(positions)-1
#dt_max
dt_max = input('Maximum frame separation in a connection (Press enter to use default value 1): ')

if(dt_max==''):
    dt_max=1

try:
    dt_max=int(dt_max)
except:
    raise ValueError('Input is not an integer')

#epsilon
epsilon = input('Weight of entropic regularization term: ')
try:
    epsilon=float(epsilon)
except:
    raise ValueError('Input is not a floating point number')

#alpha
alpha = input('Weight of measure divergence terms: ')
try:
    alpha=float(alpha)
except:
    raise ValueError('Input is not a floating point number')

log_sinkhorn = input ("Do you want to use the (slower but more stable) log domain Sinkhorn? (y/n, or enter for default yes): ")
if(log_sinkhorn=='y' or log_sinkhorn==''):
    log_sinkhorn=True
elif (log_sinkhorn=='n'):
    log_sinkhorn=False
else:
    raise ValueError("Input was not in ('y','n',enter)")

if(log_sinkhorn):
    n_steps = input ("Number of iterations for every execution of Sinkhorn's algorithm (Press enter to use default value 500): ")

    if(n_steps==''):
        n_steps=500

    try:
        n_steps=int(n_steps)
    except:
        raise ValueError('Input is not an integer')



min_weight= input ('Minimum non-neglebible weight of a conection (Press enter to use default value 1e-5): ')

if min_weight=='':
    min_weight=1e-5
try : 
    min_weight=float(min_weight)
except:
    raise ValueError('Input is not a floating point number')

output_name= input ('Name of the output file, containing the ".csv" extension (By default `filename`+"_tracked"): ')
if(output_name==''):
    loc=filename.find('.csv')
    if(loc==-1):
        raise ValueError('The input file is not a "csv"! How did we even get this far?!')
    output_name=filename[:loc]+'_tracked.csv'


print('\n\n')
class Node:#This will represent a single observation in any frame
    def __init__(self,t,idx):
        self.t=t#note that we make no distinction between time and frame
        self.idx=idx#index of that point in the list of positions (and intensities and maybe some other stuff) at time t, that we declare below. Allows access to more information about it
        
        #for track linking. Initially connected to nothing
        self.prev=self
        self.next=self
    '''
    def update_prev(self):#this ended up being redundant and should no be called anywhere
        if(self.prev==self):
            return self.prev
        self.prev=self.prev.update_prev()
        return self.prev
    def update_next(self):#same as update_prev
        if(self.next==self):
            return self.next
        self.next=self.next.update_next()
        return self.next
    '''
def try_connect(v,w):#try connectin nodes v and w if it doesn't interfere with previously created trcks
    assert(v.t<w.t)#connections only go one way as the connection weight graph is directed 
    nv=v.next
    pw=w.prev
    if(nv==v and pw==w):#so v is the end of a track and w a beginning of one
        v.next=w
        w.prev=v
        return True
    #note that in the current form we only allow for any node to have degree <=2. If you don't want that to be the case,
    #just connect things whenever their time numbers differ but then you need to prevent connections v->w->z, v-->z to existing simultaneously
    #This should be possible to resolve by checking if the two points already are connected somehow, before inserting a new track
    #A disjoint set structure is already implemented and commented out in Node so this should be very easy
    return False

class Connection:#this will allows us to sort connections later and pick the ones with highest weight
    def __init__(self,src,trg,val):
        self.src=src
        self.trg=trg#both instances of Node
        self.v=val#weight of a connection, the larger the better
        
#create matrices with connection weights. The POT library has done the messy part for us
#If you want more control over the distance matrix, there is also a function in Scipy
#print("Generating transport cost matrices")
costs=[[] for i in range(len(positions)-1)]
for t,pos in enumerate(tqdm(list(positions),ascii=True,desc='Generating cost matrices')):
    dt_range=range(1,min(dt_max+1,t_max-t+1))
    for dt in dt_range:
        costs[t].append((1/dt)*ot.dist(positions[t],positions[t+dt]))

transports = []#list of lists of transport matrices. So transports[i] will have (in most cases) size dt_max and contain as many transport matrices
for t, C in enumerate(tqdm(list(costs),ascii=True,desc='Computing transport matrices')):
    t_trans = []
    for dtm1, c in enumerate(C):
        #Using Gabriel Peyre's implementation of Sinkhorn's algorithm with slight modificaitons
        if(log_sinkhorn): 
            f_init=np.zeros(intensities[t].size)
            T=unb_Sinkhorn(c,np.array(intensities[t]).reshape((-1,1)),np.array(intensities[t+dtm1+1]).reshape((1,-1)),epsilon,alpha,f_init,niter=n_steps)
            
        #This is not implemented in log domain and so tends to blow up much more easily. However, it's also way faster
        else:
            T=ot.unbalanced.sinkhorn_unbalanced(intensities[t],intensities[t+dtm1+1],c,epsilon,alpha,verbose=True)        
        
        t_trans.append(T)
    transports.append(t_trans)

#weight assigned to a particular connection between nodes
def weight(n1,n2):
    T=transports[n1.t][n2.t-n1.t-1]
    if float('NaN') in T.flatten():
        raise ValueError
    #print(T.shape)
    if np.sum(T[n1.idx])<1e-6:
        return 0
    w=(T[n1.idx][n2.idx]/np.sum(T[n1.idx]))*2**(-(n2.t-n1.t+1))  #transport ratio with some extra penalty for large time differences
    
    return w


nodes=[[Node(t,i)for i,p in enumerate(positions[t])] for t in range(len(positions))]#we create a list of nodes to be connected
conn_list=[]

for t,nt in enumerate(tqdm(list(nodes),ascii=True,desc='Creating possible connections')):
    for i,n1 in enumerate(nt):
        for dt in range(1,min(t_max-t+1,dt_max+1)):
            for j,n2 in enumerate(nodes[t+dt]):
                w=weight(n1,n2)
                if(w>min_weight):
                    conn_list.append(Connection(n1,n2,w))

conn_list.sort(key = lambda x : x.v, reverse=True)#sort by decresing weight

n_conn=len(conn_list)#number of sufficiently large connections
    
for i in tqdm(range(n_conn),ascii=True,desc='Connecting tracks'):
    v=try_connect(conn_list[i].src,conn_list[i].trg)
    if v is False:#left here for troubleshooting
        pass
        #print(conn_list[i].src.t,conn_list[i].src,conn_list[i].trg)





#From all the relations between nodes, we want to go to a simple list of tracks
tracks=[]
for N in tqdm(nodes,ascii=True,desc='Creating a list of tracks'):
    for n in N:
        if n.prev==n and n.next!=n:
            tr=[n]
            while n.next!=n:
                n=n.next
                tr.append(n)
            tracks.append(tr)

tracks.sort(key = lambda x : len(x), reverse=True)

track_series=[]
for i,tr in enumerate(tqdm(tracks,ascii=True,desc='Creating output csv')):
    for pt in tr:
        d={'track_id' : i}
        d.update(frames[pt.t][pt.idx])
        track_series.append(pd.Series(data=d,index=d.keys()))

output_df=pd.DataFrame(track_series)
output_df.to_csv(output_name,index=False)
