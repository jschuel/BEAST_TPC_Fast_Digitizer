'''
Script reads in simulated electron recoil primary tracks that were generated with DEGRAD, or raw simulated nuclear recoil primary tracks 
that were generated with SRIM, and creates event-level pandas dataframes that are saved as .feather files and can be passed into 
the digitizer module (digitize.py). testFile.feather is an example output file from this script. This script also has functionality 
to rotate and reflect simulated primary tracks to augment primary track samples that may later be passed into 3D convolutional
neural network-based classification algorithms.
'''

import ROOT
import numpy as np
import pandas as pd
import root_pandas as rp
import os
from os import sys
from tqdm import tqdm
tqdm.pandas()
from numba import jit

class produce_tracks:
    def __init__(self, input_file, output_file, file_type, truth_dir = np.array([1,0,0]), treename = "recoils", W = 35, rotate = False, save = True):

        #Global variables
        self.input_file  = input_file
        self.file_type   = file_type #Must be 'srim' or 'degrad'
        self.output_file = output_file
        self.truth_dir   = truth_dir #Truth direction of the primary recoil track. SRIM by default produces tracks parallel to the +x axis
        self.treename    = treename  #Name of ROOT tree in SRIM file
        self.W           = W         #Average energy per ion pair in eV
        self.rotate      = rotate    #Boolean flag on whether we rotate the read in tracks or not

        #Methods
        
        if self.file_type.lower() == 'srim':
            self.tracks = self.read_and_process_SRIM()
        elif self.file_type.lower() == 'degrad':
            self.tracks = self.read_and_process_degrad()
        else:
            raise ValueError("file_type must be either 'SRIM' or 'Degrad'! Please try again.")

        self.tracks.index = [i for i in range(0,len(self.tracks))]
        
        if self.rotate: #Rotate primary track coordinates
            print("\nRandomly rotating tracks\n")
            xshift   = []
            yshift   = []
            zshift   = []
            dirshift = []
            reflect_indices = np.random.randint(0,4,len(self.tracks)) #generate random integer between 0 and 3 to determine which axis, if any, we reflect the primary track about
            for i in tqdm(range(0,len(self.tracks))):
                track = self.tracks.iloc[i]
                xsh, ysh, zsh, dirsh = self.rotate_track(track,init_dir = self.truth_dir,reflection = reflect_indices[i]) #Rotate track coordinates..also centers tracks
                xshift.append(xsh)
                yshift.append(ysh)
                zshift.append(zsh)
                dirshift.append(dirsh)
            self.tracks['chipx'] = xshift
            self.tracks['chipy'] = yshift
            self.tracks['chipz'] = zshift
            self.tracks['truth_vec'] = dirshift 
        else:
            self.tracks['truth_vec'] = [self.truth_dir for i in range(0,len(self.tracks))]
            self.tracks['chipx'] = self.tracks['chipx'].apply(lambda x: x- x.mean()) #center track regardless of whether we rotate or not
            self.tracks['chipy'] = self.tracks['chipy'].apply(lambda x: x- x.mean()) #center track
            self.tracks['chipz'] = self.tracks['chipz'].apply(lambda x: x- x.mean()) #center track

        self.process_headtail(self.tracks) #Process vector directional head/tail assignment
        
        if save:
            self.tracks.to_feather(self.output_file)
        
    def read_and_process_SRIM(self): #create a dataframe of primary track coordinates
        f = ROOT.TFile(self.input_file, 'read')
        mytree=f.Get(self.treename)
        nentry = mytree.GetEntries()
        df = pd.DataFrame()
        xs = []
        ys = []
        zs = []
        energies = []
        
        for n in tqdm(range(0,nentry)): #only loop through track-coordinates of all events
            event = mytree.GetEntry (n)    
            x = []
            y = []
            z = []
            for electron in mytree.electron_x:
                x += [0.1*electron.X()]
                y += [0.1*electron.Y()]
                z += [0.1*electron.Z()]
            energies.append(int(mytree.EkeV))
            xs.append(np.array(x).astype('float32'))
            ys.append(np.array(y).astype('float32'))
            zs.append(np.array(z).astype('float32'))
    
        df['truth_energy'] = energies
        df['chipx'] = xs
        df['chipy'] = ys
        df['chipz'] = zs
        df['NbEle'] = df['chipx'].apply(lambda x: np.array([1 for i in x]).astype('uint8'))
        df['nHits'] = df['chipx'].apply(lambda x: len(x))
        df['primary_track_energy'] = df['NbEle'].apply(lambda x: x.sum())*self.W/1000.
        df['IQF'] = df['primary_track_energy']/df['truth_energy']
        
        return df

    def read_and_process_degrad(self): #DEGRAD files are root files. For convenience, we use root_pandas to open them
        df = rp.read_root(self.input_file)
        df = df.rename(columns={"npoints": "nHits", "x": "chipx", "y": "chipy", "z": "chipz",
                           "q": "NbEle"})
        df['primary_track_energy'] = df['NbEle'].apply(lambda x: x.sum())*self.W/1000.
        return df

    def rotate_track(self,track,init_dir,reflection):
        
        def random_theta_phi(): #get random theta and phis for rotation
            ctheta = np.random.uniform(-1,1) #draw from uniform cos(theta) distribution
            phi = np.random.uniform(0,2*np.pi)
            theta = np.arccos(ctheta)
            x = np.sin(theta)*np.cos(phi)
            y = np.sin(theta)*np.sin(phi)
            z = np.cos(theta)
            return ROOT.TVector3(x,y,z).Theta(), ROOT.TVector3(x,y,z).Phi()

        def rotate_y(x,y,z,angle): #rotate about y axis
            xp = np.cos(angle)*x+np.sin(angle)*z
            yp = y
            zp = -np.sin(angle)*x+np.cos(angle)*z
            return xp,yp,zp
    
        def rotate_z(x,y,z,angle): #rotate about z axis
            xp = np.cos(angle)*x-np.sin(angle)*y
            yp = np.sin(angle)*x+np.cos(angle)*y
            zp = z
            return xp,yp,zp

        ###Reflect tracks to make rotations overall more general###
        if reflection == 0: 
            xs = track['chipx']
            ys = track['chipy']
            zs = track['chipz']
        
        elif reflection == 1:
            xs = -1*track['chipx']
            ys = track['chipy']
            zs = track['chipz']
            init_dir = np.array([-1*init_dir[0],init_dir[1],init_dir[2]]) 
        elif reflection == 2:
            xs = track['chipx']
            ys = -1*track['chipy']
            zs = track['chipz']
            init_dir = np.array([init_dir[0],-1*init_dir[1],init_dir[2]]) 
        else:
            xs = track['chipx']
            ys = track['chipy']
            zs = -1*track['chipz']
            init_dir = np.array([init_dir[0],init_dir[1],-1*init_dir[2]]) 
    
        theta, phi = random_theta_phi() #get different random values for each instance

        ### Rotate tracks to make them directionally isotropic ###
        x_shift1, y_shift1, z_shift1 = rotate_y(xs, ys, zs,-(np.pi/2-theta)) #rotate track coordinates about y axis
        x_shift2, y_shift2, z_shift2 = rotate_z(x_shift1, y_shift1, z_shift1,phi) #rotate track coordinates about z axis
        dir_shiftx1, dir_shifty1, dir_shiftz1 = rotate_y(init_dir[0], init_dir[1], init_dir[2], -(np.pi/2-theta)) #rotate reocil direction about y axis
        dir_shiftx2, dir_shifty2, dir_shiftz2 = rotate_z(dir_shiftx1, dir_shifty1, dir_shiftz1, phi) #rotate reocil direction about z axis
        
        dir_shift = np.array([dir_shiftx2, dir_shifty2, dir_shiftz2])

        #center tracks
        xshifted = x_shift2-x_shift2.mean()
        yshifted = y_shift2-y_shift2.mean()
        zshifted = z_shift2-z_shift2.mean()
        
        return xshifted, yshifted, zshifted, dir_shift

    def process_headtail(self,df): #Compares primary track direction with true recoil direction.
        # If the scalar (dot) product of the primary track's principal axis and the true recoil direction is positive,
        # then the vector "head/tail" direction of the track is correct
        
        @jit(nopython=True)
        def get_PA(data):
            uu, dd, vv = np.linalg.svd(data-np.array([data[:,0].mean(),data[:,1].mean(),data[:,2].mean()]))
            projection = (data @ vv.T).T[0]
            return projection, vv[0]

        def apply_headtail(vec, cf):
            if cf > 0.5:
                vec = -vec
                cf = 1-cf
            elif cf < 0.5:
                pass
            else:
                rand = np.random.randint(0,2)
                if rand == 1:
                    vec = -vec
                    cf = 1-cf
                else:
                    pass
            return vec, cf

        projectt = [] #projection along primary track's principal axis
        vecst    = [] #primary track principal axis vectors
        
        print("\nIdentifying principal axis.\n")

        for i in tqdm(range(0,len(df))):
            truth = np.concatenate([[df['chipx'][i].T,df['chipy'][i].T,df['chipz'][i].T]]).T
            projt, vect = get_PA(truth)
            projectt.append(projt)
            vecst.append(vect)
        df['proj_truth'] = projectt
        df['vec_truth'] = vecst

        print("\nComputing track charge fractions.\n")

        df['mid_point'] = (df['proj_truth'].apply(lambda x: x.max()) + df['proj_truth'].progress_apply(lambda x: x.min()))/2
        df['CF'] = [df['NbEle'].iloc[i][np.where(df['proj_truth'].iloc[i]>df['mid_point'].iloc[i])[0]].sum()/df['NbEle'].iloc[i].sum() for i in tqdm(range(0,len(df)))]

        print("\nUsing head-charge-fraction to perform head/tail corrections.\n")
        vecs = []
        cfs = []
        for i in tqdm(range(0,len(df))):
            vec = df['vec_truth'].iloc[i]
            cf = df['CF'].iloc[i]
            vec, cf = apply_headtail(vec,cf)
            vecs.append(vec)
            cfs.append(cf)
        df['vec_truth'] = vecs
        df['CF'] = cfs
        
        print("\nComparing with truth recoil direction to determine correct head/tail assignment.\n")
        df['dots'] = [np.dot(df['vec_truth'][i],df['truth_vec'][i]) for i in tqdm(range(0,len(df)))]
        df['correct_assign'] = 1
        index = df.query('dots < 0').index.to_numpy()
        df['correct_assign'][index] = 0
        
if __name__ == "__main__":
    
    a = produce_tracks(input_file = 'data/degradSample.root', output_file = 'data/to_digitize/degradSample_processed.feather', file_type = 'Degrad', rotate = False)
    
