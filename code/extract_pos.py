import pandas as pd
from matplotlib import pyplot as plt

#extracts positions from the standard thunderstorm output format
def import_pos(t_start,t_end,path,px_size):
    df=pd.read_csv(path)

    df.rename(columns = {'frame' : 't', 'x [nm]' : 'x','y [nm]' : 'y','intensity [photon]' : 'intensity'}, inplace=True)
    df.t-=1#because frames are numbered from 1 and times from 0
    df=df.loc[(df.t>=t_start) & (df.t<=t_end)]

    #preprocessing
    df.x/=px_size
    df.y/=px_size

    split_frames= [d for _,d in df.groupby(df.t)]
    int_ranges=[[d.intensity.min(),d.intensity.max()]for d in split_frames]

    for i,row in df.iterrows():
        df.loc[i,'intensity']=(row.intensity)/(int_ranges[int(row.t)][1])
    
    return df