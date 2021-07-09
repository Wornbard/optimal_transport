import numpy as np
import pandas as pd

class particle_system:
    def __init__(self,grid):#grid:integer tuple
        self.grid=grid
        self.particles=[]
        
    def insert_particles(self,n,update_pos,update_intensity,start_pos=None,intensity=None):
        if start_pos is None:
            x_data = np.random.uniform(low=0, high=self.grid[0], size=n)
            y_data= np.random.uniform(low=0, high=self.grid[1], size=n)
            start_pos=[(x_data[i],y_data[i])for i,x in enumerate(x_data)]
            
        if intensity is None:
            intensity=[1 for i in range(n)]
            
        for i in range(n):
            self.particles.append(particle(self.grid,start_pos[i],update_pos,intensity[i],update_intensity))
    def evolve(self,dt):
        for p in self.particles:
            p.evolve(dt)
    def positions(self):
        return [p.pos for p in self.particles]
    def intensities(self):
        return [p.intensity for p in self.particles]


class particle:
    def __init__(self,grid,start_pos,update_pos,intensity,update_intensity):
        self.grid=grid#dimensions of the space it's restricted to
        
        self.pos=start_pos#current position
        self.__class__.update_pos=update_pos
        
        self.intensity=intensity#current intensity
        self.__class__.update_intensity = update_intensity
        
    def evolve(self,dt):
        self.pos=self.update_pos(dt)
        self.intensity=self.update_intensity(dt)


def inside_grid(pos,grid):#assume grid has left bottom corner in (0,0), check if a given point is inisde of it
    return pos[0]>=0 and pos[0]<=grid[0] and pos[1]>=0 and pos[1]<=grid[1]


def brownian_update(part,dt,D,drift=[0,0]):
    mean=[dt*d for d in drift]
    cov=np.identity(2)*dt*D*2#the 2 term appears because I think that's the 'true' variance formula in Brownian motion but D may as well just be any number, not the diffusion coeff
    dx, dy = np.random.multivariate_normal(mean, cov, 1).T
    dx=dx[0]
    dy=dy[0]
    if (dx>part.grid[0] or dy>part.grid[1]):
        print("The time step in brownian motion seems to be too large relative to grid size")#just to see if I'm doing anything stupid
    pos=part.pos
    new_pos=tuple(map(sum, zip(pos, (dx,dy))))
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
        pos.append((row[1][0],row[1][1]))
        intensity.append(row[1][2])
    return pos,intensity