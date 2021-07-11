import numpy as np
import pandas as pd

class particle_system:
    def __init__(self,grid):#grid:integer tuple
        self.grid=grid
        self.particles=[]
        self.dim=len(self.grid)
        
    def insert_particles(self,n,update_pos,update_intensity,start_pos=None,intensity=None):
        if start_pos is None:
            pos_data=[np.random.uniform(low=0, high=self.grid[i], size=n) for i in range(self.dim)]
            start_pos=[[pos_data[j][i] for j in range(self.dim)]for i in range(n)]
        if intensity is None:
            intensity=[1 for i in range(n)]
            
        for i in range(n):
            self.particles.append(particle(self.grid,self.dim,start_pos[i],update_pos,intensity[i],update_intensity))
    def evolve(self,dt):
        for p in self.particles:
            p.evolve(dt)
    def positions(self):
        return [p.pos for p in self.particles]
    def intensities(self):
        return [p.intensity for p in self.particles]


class particle:
    def __init__(self,grid,dim,start_pos,update_pos,intensity,update_intensity):
        self.grid=grid#dimensions of the space it's restricted to
        self.dim=dim
        
        self.pos=start_pos#current position
        self.__class__.update_pos=update_pos
        
        self.intensity=intensity#current intensity
        self.__class__.update_intensity = update_intensity
        
    def evolve(self,dt):
        self.pos=self.update_pos(dt)
        self.intensity=self.update_intensity(dt)

def inside_grid(pos,grid):#assume grid has leftmost bottom corner in 0, check if a given point is inisde of it
    inside_dimensions=[pos[i]>=0 and pos[i]<=grid_dim for i,grid_dim in enumerate(grid)]
    return all(inside_dimensions)

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