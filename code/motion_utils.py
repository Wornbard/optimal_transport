import numpy as np
import pandas as pd

class particle_system:
    '''
    Keep's track of the system's size, state and the particels in it
    Constructor takes a list of length equal to the system's dimension
    For [d1,d2,...,dk] the domain on which the particle can move is [0,d1]x[0,d2]x ... x[0,dk]
    '''
    def __init__(self,grid,creation):#grid : array of length equal to dimension
        self.grid=grid
        self.particles=[]
        self.dim=len(self.grid)
        self.max_id=0
        self.__class__.creation=creation
    '''
    n : number of particles to insert
    update_pos : a function which is passed to the constructor of particle, used for evolving it in time (see description of particle class for details)
    update_intensity : as above
    start_pos : if None, then generates randomly, otherwise a list of length n containing the initial positions as lists of length self.dim
    intensity : if None, then set to 1, otherwise a list of length n

    '''
    def insert_particles(self,n,update_pos,update_intensity,start_pos=None,intensity=None):
        if start_pos is None:
            pos_data=[np.random.uniform(low=0, high=self.grid[i], size=n) for i in range(self.dim)]
            start_pos=[[pos_data[j][i] for j in range(self.dim)]for i in range(n)]
        if intensity is None:
            intensity=[1 for i in range(n)]
            
        for i in range(n):
            self.particles.append(particle(self.grid,start_pos[i],update_pos,intensity[i],update_intensity,id=self.max_id+i))
        self.max_id+=n
    
    '''
    updates the state of each particle in the system
    '''
    def evolve(self,dt):
        for p in self.particles:
            p.evolve(dt)
            if p.intensity==0:#remove if intensity drops to 0
                self.particles.remove(p)
        self.creation(dt)

    def positions(self):
        return [p.pos for p in self.particles]
    def intensities(self):
        return [p.intensity for p in self.particles]


class particle:
    '''
    grid : as in particle_system
    start_pos : initial position in a list
    update_pos : assigned as a method of the class so the arguments it takes are : instance of class particle, time increment
    intenisty : initial intensity
    update_intsnity : as in update_pos. Note that the current method to make a particle vanish is to set its intensity to 0 so this method should handle that
    '''

    def __init__(self,grid,start_pos,update_pos,intensity,update_intensity,id):
        self.grid=grid#dimensions of the space it's restricted to
        self.dim=len(self.grid)
        
        self.pos=start_pos#current position
        self.__class__.update_pos=update_pos
        
        self.intensity=intensity#current intensity
        self.__class__.update_intensity = update_intensity
        
        self.id=id
    def evolve(self,dt):
        self.pos=self.update_pos(dt)
        self.intensity=self.update_intensity(dt)

'''
checks if a point is contained in a grid
'''
def inside_grid(pos,grid):#assume grid has leftmost bottom corner in 0, check if a given point is inisde of it
    inside_dimensions=[pos[i]>=0 and pos[i]<=grid_dim for i,grid_dim in enumerate(grid)]
    return all(inside_dimensions)


'''
brownian motion of a single particle in n dimensions
part : particle
dt: time increment
D : diffusion coefficient
drift : list of length equal to dimension if we want to have a drift in the motion

Note that when passing this to a constructor we need to specify the D and drift parameter with a lambda i.e.
pass lambda x,y : brownian_update(x,y,some_D,some_drift)
'''
def brownian_update(part,dt,D,drift=None):
    if drift is None:
        drift=[0]*part.dim
    mean=[dt*d for d in drift]
    cov=np.identity(part.dim)*dt*D*2#the 2 term appears because I think that's the 'true' variance formula in Brownian motion but D may as well just be any number, not the diffusion coeff
    dr = np.random.multivariate_normal(mean, cov, 1).T
    if (any([np.abs(dr[i][0])>grid_dim/2 for i,grid_dim in enumerate(part.grid)])):#this is just bad code but we're unlikely to need more dimensions anytime soon
        print("The time step in brownian motion seems to be too large relative to grid size")#just to see if I'm doing anything stupid
    pos=part.pos
    new_pos=list(map(sum, zip(pos, [dr[i][0] for i,d in enumerate(dr)])))
    if inside_grid(new_pos,part.grid):
        return new_pos
    else:#need to figure out where and when it intersects the boundary. For now let's just resample
        return brownian_update(part,dt,D,drift)


'''
Takes an output file of thunderstorm in csv, frame number (starting from 1), and returns two lists of positions and intensities of every observed particles in a given frame
'''
def thunderstorm_extract(directory,frame_id):
    df=pd.read_csv(directory)
    frame=df.loc[df['frame'] == frame_id]
    data=frame[['x [nm]','y [nm]','intensity [photon]']]
    pos=[]
    intensity=[]
    for row in data.iterrows():
        pos.append([row[1][0],row[1][1]])
        intensity.append(row[1][2])
    return pos,intensity


def placeholder_intensity_update(part,dt,mean,variance=None,threshold=None):
    if variance is None:
        variance=(dt)**2
    if threshold is None:
        threshold=0.1*mean#if intensity drops below then it disappears
    #print(mean,variance,part.intensity)
    intensity_change=np.random.normal(mean-part.intensity, variance)
    new_int=part.intensity+intensity_change
    #print(part.intensity)
    if new_int<threshold:
        new_int=0
    return new_int

