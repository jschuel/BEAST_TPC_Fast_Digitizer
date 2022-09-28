'''
Module to simulate event generation in a time projection chamber. This module is fed in processed primary tracks that were generated using
produce_primary_tracks.feather. The module will (1) read in the primary track, (2), drift the track over a specified drift length. The
charges in the track will diffuse proportionally to sqrt(drift_length) with transverse and longitudinal diffusion strengths controlled
by 'sigmaT' and 'sigmaL', respectively. After diffusion the track will be (3) amplified following a random exponential distribution with the
scale determined by the 'gain' parameter, and an additional smearing is applied, with the strength of the smearing determined by 'sigma{T,L}e'. 
The amplified charge will then be binned into a grid of dimensions specified by the '{col,row,z}_{length,bins}' set of parameters. 
A hit will be registered in a bin if the charge + noise of the hit is greater than the 'threshold'.The charge quantization of our readout 
chip is given by the 'pixel_charge' mapping, which was determined using measured data in our lab.
'''

import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import jit
import ROOT
from ROOT import TVector3
from os import sys
import h5py as h5
import hdf5plugin
pd.set_option('mode.chained_assignment', None) #remove pandas copy warning

class digitize:
    def __init__(self, production = 'SRIM', col_length = 2, col_bins = 80, row_length = 1.68, row_bins = 336, z_length = 2.5, z_bins = 100, threshold = 3137, gain = 13000, W = 35, drift_length = 2.5, sigmaT = 134.8, sigmaL = 128.2, sigmaTe = 143, sigmaLe = 97, pixel_charge = np.array([3119, 3573, 5625, 8659, 11373, 14471, 17912, 21744, 25655, 29658, 33682, 37944, 42483, 50104]),tracks_file = 'data/to_digitize/testFile.feather', output_file = 'data/digitized/testFileDigitized.feather', noise_file = 'data/sampleNoise.h5', randomize_location = True, apply_noise = True, save = True):

        self.production = production #SRIM, DEGRAD or G4
        self.col_length = col_length #Chip length in x (cm)
        self.col_bins = col_bins     #x-segmentation
        self.col_width = self.col_length/self.col_bins #width of each pixel in x
        self.row_length = row_length #Chip length in y (cm)
        self.row_bins = row_bins     #y-segmentation
        self.row_width = self.row_length/self.row_bins #width of each pixel in y
        self.z_length = z_length     #Length of BCID window
        self.z_bins = z_bins         #BCID segmentation
        self.z_width = self.z_length/self.z_bins #width of each BCID
        self.threshold = threshold   #threshold in units of electron charge
        self.gain = gain             #gain
        self.W = W                   #average energy per electron-ion pair
        self.drift_length = drift_length #For SRIM this should be set (value in cm). For G4, this parameter is ignored.
        self.sigmaT= sigmaT          #transverse diffusion coefficient during drift (um/sqrt(cm))
        self.sigmaL = sigmaL         #longitudinal diffusion coefficient during drift (um/sqrt(cm))
        self.sigmaTe = sigmaTe       #transverse readout plane resolution (ignoring FE-I4) (um/sqrt(cm))
        self.sigmaLe = sigmaLe       #longitudinal readout plane resolution (ignoring FE-I4) (um/sqrt(cm))
        self.pixel_charge = pixel_charge #TOT to charge mapping (entries in units of electron charge)
        self.tracks_file = tracks_file #path to file iwth primary tracks to be digitized
        self.output_file = output_file #output file name. Only used if save = True
        self.noise_file = noise_file   #Output threshold scan file to determine the noise floor. Only applicable if apply_noise = True.
        
        self.tracks = self.read_tracks(self.production) 
        #self.tracks = self.tracks[0:10] #uncomment if you want to test on a subsample of tracks
        
        self.randomize_location = randomize_location #If True, randomizes the location of the amplified event along the chip to fairly sample from the noise floor. Only works well if track is fiducialized
        self.apply_noise = apply_noise #Applies a noise floor so that a pixel hit will only register if charge + noise of pixel > threshold
        
        xdiff      = []
        ydiff      = []
        zdiff      = []
        column     = []
        row        = []
        BCID       = []
        tot        = []
        q          = []
        q_truth    = []
        fiducialxy = []
        fiducialz  = []
        
        if self.apply_noise:
            self.base_noise = self.compute_base_noise() #only compute this once
        else:
            self.base_noise = np.zeros((336,80)) #not needed but makes code cleaner
        
        for i in tqdm(range(0,len(self.tracks))):
            track = self.tracks.iloc[i]
            track = self.apply_diffusion(track)
            post_gain = self.GEM_gain_and_diffusion(track)
            
            if self.apply_noise:
                noise = np.random.normal(0,self.base_noise) #create different noise distribution for each event
            else:
                noise = self.base_noise
                
            try: #Method should work unless there's no charge above threshold after amplification
                digi = self.digitize_fei4(post_gain[0],post_gain[1],post_gain[2],noise)
                column.append(digi[0])
                row.append(digi[1])
                BCID.append(digi[2])
                tot.append(digi[3])
                q.append(digi[4])
                q_truth.append(digi[5])
                fiducialxy.append(digi[6])
                fiducialz.append(digi[7])
                
            except: #Append -1's and delete track in the process_events() method. This should only occur for tracks with no hits above the readout charge threshold
                column.append(np.array([-1]))
                row.append(np.array([-1]))
                BCID.append(np.array([-1]))
                tot.append(np.array([-1]))
                q.append(np.array([-1]))
                q_truth.append(np.array([-1]))
                fiducialxy.append(np.array([-1]))
                fiducialz.append(np.array([-1]))
                
        self.tracks['column'] = column
        self.tracks['row'] = row
        self.tracks['BCID'] = BCID
        self.tracks['tot'] = tot
        self.tracks['pixel_charge'] = q
        self.tracks['truth_q'] = q_truth
        self.tracks['fiducial_xy'] = fiducialxy
        self.tracks['fiducial_z'] = fiducialz
        
        print("Processing %s of %s events"%(len(self.tracks.dropna()),len(self.tracks)))
        self.process_events()

        if save:
            self.tracks.to_feather(self.output_file)
            
    def read_tracks(self, prod):
        if prod.lower() == 'srim' or prod.lower() == 'degrad': #Currently both options perform identical tasks, this may change in the future
            df = pd.read_feather(self.tracks_file)
            df['chipz'] = df['chipz']+self.drift_length #z is 0 in the default files produced, so we shift primary tracks upward for drift
        else:
            raise ValueError("Production argument must be SRIM or Degrad")
        return df

    def apply_diffusion(self, track): #applies diffusion using input diffusion parameters
        xs = []
        ys = []
        zs = []
        for x, y, z, e in zip(track['chipx'],track['chipy'],track['chipz'],track['NbEle']):
            for j in range(0,int(np.round(e,0))):
                xs.append(x)
                ys.append(y)
                zs.append(z)
        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)
        x_diff = np.sqrt(zs)*self.sigmaT*1e-4*np.random.normal(0,1, len(zs))
        y_diff = np.sqrt(zs)*self.sigmaT*1e-4*np.random.normal(0,1, len(zs))
        z_diff = np.sqrt(zs)*self.sigmaL*1e-4*np.random.normal(0,1, len(zs))
        xs = xs+x_diff
        ys = ys+y_diff
        zs = zs+z_diff
        track['x'] = xs
        track['y'] = ys
        track['z'] = zs
        
        del x_diff, y_diff, z_diff
        return track

    def generate_gain_points(self, x, x_post, gain_electrons, sigma = 'Transverse'): #Generates x, y, and z coordiantes after gain
        if sigma.lower() == 'transverse':
            sig = self.sigmaTe
        else:
            sig = self.sigmaLe
        for enum, val in np.ndenumerate(gain_electrons):
            start_ind = np.sum(gain_electrons[:enum[0]])
            end_ind = np.sum(gain_electrons[:enum[0]+1])
            x_post[start_ind:end_ind] = x[enum] + sig*1E-4*np.random.normal(0,1,val)

    def GEM_gain_and_diffusion(self,track): #Applies gain and readout resolution smearing
        gain_electrons = np.random.exponential(self.gain, len(track['x']))
        gain_electrons = np.asarray(gain_electrons, dtype=int)
        
        x_post = np.ascontiguousarray(np.zeros(np.sum(gain_electrons)),dtype=np.float32)
        y_post = np.ascontiguousarray(np.zeros(np.sum(gain_electrons)),dtype=np.float32)
        z_post = np.ascontiguousarray(np.zeros(np.sum(gain_electrons)),dtype=np.float32)
        
        self.generate_gain_points(track['x'].astype('float32'), x_post, gain_electrons, sigma='Transverse')
        self.generate_gain_points(track['y'].astype('float32'), y_post, gain_electrons, sigma='Transverse')
        self.generate_gain_points(track['z'].astype('float32'), z_post, gain_electrons, sigma='Longitudinal')

        return x_post, y_post, z_post

    def compute_base_noise(self): #computes noise floor from threshold scan
        f = h5.File(self.noise_file,'r')
        dfm = pd.read_hdf(self.noise_file,key = 'meta_data')
        noise = f['HistNoiseFittedCalib'][:]
        return noise

    def digitize_fei4(self,x,y,z,noise): #produce_primary_tracks.py centers the tracks so we draw the chip with a center of (0,0)

        #Center binning window at (0,0)
        x_bins = np.linspace(-self.col_length/2, self.col_length/2, self.col_bins+1)
        y_bins = np.linspace(-self.row_length/2, self.row_length/2, self.row_bins+1)
        z_bins = np.linspace(0, self.z_length,   self.z_bins+1) #BCID is always positive

        if self.randomize_location: #randomizes (x,y) location to fairly sample noise floor
            
            #These xshift and yshifts correspond to bins centered at (0,0)
            xshift = np.random.uniform(-(self.col_length/2+x.min()), self.col_length/2-x.max())
            yshift = np.random.uniform(-(self.row_length/2+y.min()), self.row_length/2-y.max())
            x = x+xshift
            y = y+yshift

        #fiducialxy checks if all charge is within the fiducial area of the chip. value is 1 if True and 0 if False
        
        fiducialxy = int(x.max() <= self.col_length/2 and x.min() >= -self.col_length/2 and y.max() <= self.row_length/2 and y.min() >= -self.row_length/2)
        fiducialz  = int((z.max()-z.min()) <= self.z_length) #Checks if z extent goes beyond BCID_window. 1 if True, 0 if False
        
        hits = np.array([x,y,z-z.min()]).T

        pix = np.histogramdd(hits, bins=(x_bins, y_bins, z_bins))[0]
        flat_pix = np.sum(pix, axis=2)
        
        row, column = np.where(flat_pix.T + noise > self.threshold)
        q = flat_pix[column, row].flatten()+noise.T[column,row].flatten()

        def find_tot(arr,tot_map): #compute TOT from charge
            tots = []
            for charge in arr:
                for i in range(1,len(tot_map)):
                    if charge > tot_map[i] and i < 13:
                        continue
                    elif charge < tot_map[i]:
                        tots.append(i-1)
                        break
                    elif charge == tot_map[i]:
                        tots.append(i)
                        break
                    else:
                        tots.append(13)
            return np.array(tots)
    
        tot = find_tot(q,self.pixel_charge)
        
        bcid = []
        for i in range(0,len(column)):
            bcid_over_thresh = np.where(np.cumsum(pix[column[i],row[i]]) + noise.T[column[i],row[i]] > self.threshold)[0][0]
            bcid.append(bcid_over_thresh)

        bcid = np.array(bcid)
        pixel_charge = self.pixel_charge[tot]

        return column, row, bcid, tot, pixel_charge, q, fiducialxy, fiducialz

    def process_events(self): #Computes variables of interest for digitized tracks
        
        self.tracks['npoints'] = self.tracks['column'].apply(lambda x: len(x)) #number of pixel hits
        self.tracks['x'] = self.tracks['column'].apply(lambda x: x*self.col_width*1e4)
        self.tracks['y'] = self.tracks['row'].apply(lambda x: (335-x)*self.row_width*1e4)
        self.tracks['z'] = self.tracks['BCID'].apply(lambda x: x*self.z_width*1e4)
        self.tracks['track_charge'] = self.tracks['pixel_charge'].apply(lambda x: x.sum()) #charge of entire track
        self.tracks['track_energy'] = self.tracks['track_charge']*self.W/self.gain/1000 #reconstructed energy of track

        def get_angles_and_charge_fractions(data,vec,q): #computes track angles and head/tail charge asymmetry
            Tvec = TVector3(vec[0],vec[1],vec[2])
            theta = Tvec.Theta() #RADIANS
            phi = Tvec.Phi() #RADIANS
            if np.cos(theta) < 0: #restrict vector so head points up always
                vec = -1 * Tvec
                theta = Tvec.Theta()
                phi = Tvec.Phi()
            v = np.array([Tvec.x(), Tvec.y(), Tvec.z()]) #convert back to numpy
            projection = data @ v.T
            midp = 0.5*float(projection.max()+projection.min())
            uc = 0 #upper half charge
            lc = 0 #lower half charge
            for i,val in enumerate(projection):
                if val > midp:
                    uc += q[i]
                elif val < midp:
                    lc += q[i]
                elif val == midp and i%2 == 0:
                    uc += q[i]
                elif val == midp and i%2 != 0:
                    lc += q[i]
            upper_charge_fraction = uc/(uc+lc)
                
            return theta, phi, upper_charge_fraction, uc, lc

        @jit(nopython=True) #numba fast compile
        def get_principal_axis(data):
            uu, dd, vv = np.linalg.svd(data-np.array([data[:,0].mean(),data[:,1].mean(),data[:,2].mean()]))
            projection = (data @ vv.T).T[0]
            return projection.max() - projection.min(), vv[0]

        @jit(nopython=True)
        def SDCD(data): #standard deviation of charge distribution
            return np.sqrt(np.diag(data @ data.T).mean())
    
        @jit(nopython=True)
        def wSDCD(wdata,q): #charge weighted standard deviation of charge distribution
            return np.sqrt(np.diag(wdata @ wdata.T).sum()/(q.sum()))

        @jit(nopython=True) 
        def compute_y_track_hat(zhat,xhat): #y unit vector orthogonal to principal axis of track
            return np.cross(zhat,xhat)

        @jit(nopython=True)
        def compute_z_track_hat(xhat,yhat): #z unit vector orthogonal to principal axis of track
            return np.cross(xhat,yhat)

        @jit(nopython=True)
        def compute_track(data,tuv): #computes track coordinate system; tuv is track unit vector
            return data @ tuv

        def ChargeUnif(data):  #std dev of distribution of mean distances between each charge and all other charges
            a = np.linalg.norm(data - data[:,None], axis=-1)
            return np.std([a[i].mean() for i in range(0,len(data))])

        self.tracks['sum_tot'] = self.tracks['tot'].apply(lambda x: x.sum()) #total charge of event in units of TOT
        self.tracks['sat_frac'] = self.tracks['tot'].apply(lambda x: len(np.where(x == 13)[0])/len(x)) #fraction of saturated pixels in event

        #center tracks for track coordinate computations
        self.tracks['x_center'] = self.tracks['x'] - self.tracks['x'].apply(lambda x: x.mean())
        self.tracks['y_center'] = self.tracks['y'] - self.tracks['y'].apply(lambda x: x.mean())
        self.tracks['z_center'] = self.tracks['z'] - self.tracks['z'].apply(lambda x: x.mean())

        ls = [] #lengths
        xt_hats = [] #principal axis vectors, direction of x_track_hat
        yt_hats = []
        zt_hats = []
        xtracks = [] #coordinates in track coordinate system
        ytracks = []
        ztracks = []
        stds = [] #std deviations of charge distribution
        wstds = [] #charge weighted std deviations of charge distribution
        thetas = [] #zenith angles
        phis = [] #azimuthal angles with respect to readout plane
        ucfs = [] #Fractions of total track charge on upper half of track
        ucs = [] #Charge on the upper half of the track
        lcs = [] #Charge on the lower half of the track
        chargeunif = [] #std dev of distribution of mean distances between each charge and all other charges
        
        print("Computing e rejection discriminants...")
        zhat = np.array([0,0,1],dtype='float32') #initial zhat vector
        
        for i in tqdm(range(0,len(self.tracks))): #construct tracks to store event-level information in a dataframe
            track = np.concatenate([[self.tracks['x'][i].T,self.tracks['y'][i].T,self.tracks['z'][i].T]]).T
            ctrack = np.concatenate([[self.tracks['x_center'][i].T,self.tracks['y_center'][i].T,self.tracks['z_center'][i].T]]).T #centered track
            wtrack = np.concatenate([[self.tracks['pixel_charge'][i]*self.tracks['x_center'][i].T,self.tracks['pixel_charge'][i]*self.tracks['y_center'][i].T,self.tracks['pixel_charge'][i]*self.tracks['z_center'][i].T]]).T # for wSDCD
            q = self.tracks['pixel_charge'][i]
            l, xt_hat = get_principal_axis(track) #length and principal axis vector
            ls.append(l)
            xt_hats.append(xt_hat) #principal axis direction
            yt_hat = compute_y_track_hat(zhat,xt_hat)
            ynorm = np.linalg.norm(yt_hat)
            yt_hat = yt_hat/ynorm #normalize
            yt_hats.append(yt_hat)
            zt_hat = compute_z_track_hat(xt_hat,yt_hat)
            zt_hats.append(zt_hat)
            xtrack = compute_track(ctrack,xt_hat)
            ytrack = compute_track(ctrack,yt_hat)
            ztrack = compute_track(ctrack,zt_hat)
            xtracks.append(xtrack)
            ytracks.append(ytrack)
            ztracks.append(ztrack)
            stds.append(SDCD(ctrack))
            wstds.append(wSDCD(wtrack,q))
            chargeunif.append(ChargeUnif(track))
            th, ph, ucf, uc, lc = get_angles_and_charge_fractions(track,xt_hat,q) #theta, phi, upper charge fraction, upper charge, lower charge, respectively
            thetas.append(th)
            phis.append(ph)
            ucfs.append(ucf)
            ucs.append(uc)
            lcs.append(lc)
        self.tracks['x_track'] = xtracks
        self.tracks['y_track'] = ytracks
        self.tracks['z_track'] = ztracks
        self.tracks['theta'] = thetas
        self.tracks['phi'] = phis
        self.tracks['upper_charge_fraction'] = ucfs
        self.tracks['upper_charge'] = ucs
        self.tracks['lower_charge'] = lcs
        self.tracks['LAPA'] = ls
        self.tracks['SDCD'] = stds
        self.tracks['wSDCD'] = wstds
        self.tracks['CylThick'] = (self.tracks['y_track']**2+self.tracks['z_track']**2).apply(lambda x: x.sum())
        self.tracks['ChargeUnif'] = chargeunif
        
        self.tracks = self.tracks.query('npoints > 1') #remove all tracks with 0 or 1 hit above threshold
        self.tracks.index = [i for i in range(0,len(self.tracks))] #reindex

        print('DONE!')
        
if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        raise ValueError("Gain must be specified when calling this script. For example, if you want a gain of 1,000, use 'python3 digitize.py 1000'")

    gain = sys.argv[1]
    
    tracks_file = 'data/to_digitize/testFile.feather'
    noise_file  = 'data/sampleNoise.h5'
    output_file = 'data/digitized/testFileDigitized_gain%s.feather'%(gain)

    a = digitize(production = 'SRIM', tracks_file = tracks_file, noise_file = noise_file, output_file = output_file, gain = int(gain))
