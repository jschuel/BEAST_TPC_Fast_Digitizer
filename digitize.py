import numpy as np
import pandas as pd
from ROOT import TVector3
from tqdm import tqdm
import os
pd.set_option('mode.chained_assignment', None) #remove pandas copy warning
from numba import jit
import warnings
warnings.filterwarnings('ignore') #Used to ignore numba warnings. Comment out when debugging

class digitize:
    def __init__(self, col_width = .025, row_width = .0050, z_width = .0250, threshold = 3137, gain = 15000, sigmaT = 134.8, sigmaL = 128.2, sigmaTe = 143, sigmaLe = 97, W = 0.035, pixel_charge = np.array([3237, 3253.2, 3681.4, 4936.4, 7224.8, 8972.2, 12143.7, 16204, 20159, 24351, 28696, 33368, 38931, 52141]),tracks_file = 'test_to_digitize.feather', post_process = True, save = True): #Digitization parameters are arguments to the digitize class
        
        self.col_width = col_width
        self.row_width = row_width
        self.z_width = z_width
        self.threshold = threshold
        self.gain = gain
        self.sigmaT= sigmaT
        self.sigmaL = sigmaL
        self.sigmaTe = sigmaTe
        self.sigmaLe = sigmaLe
        self.pixel_charge = pixel_charge
        #Compute midpoint of every other point in pixel_charge array. Needed to get correct TOT code when digitizing
        self.pixel_charge_mid = np.array([(pixel_charge[i] + pixel_charge[i+1])/2 for i in range(0,len(pixel_charge)-1)])
        self.tracks_file = tracks_file
        self.tracks = self.read_tracks()
        xdiff = []
        ydiff = []
        zdiff = []
        column = []
        row    = []
        BCID   = []
        tot    = []
        q      = []
        for i in tqdm(range(0,len(self.tracks))):
            track = self.tracks.iloc[i]
            track = self.apply_diffusion(track)
            post_gain = self.GEM_gain_and_diffusion(track)
            #try:
            digi = self.digitize_fei4(post_gain[0],post_gain[1],post_gain[2])
            column.append(digi[0])
            row.append(digi[1])
            BCID.append(digi[2])
            tot.append(digi[3])
            q.append(digi[4])
        self.tracks['column'] = column
        self.tracks['row'] = row
        self.tracks['BCID'] = BCID
        self.tracks['tot'] = tot
        self.tracks['pixel_charge'] = q
        self.tracks['pixel_energy'] = self.tracks['pixel_charge']*W/self.gain
        self.tracks['x'] = self.tracks['column'].apply(lambda x: x*self.col_width*10000.) #250um x-pitch
        self.tracks['y'] = self.tracks['row'].apply(lambda x: (335-x)*self.row_width*10000) #50um y-pitch, 335-x makes the coordinate system right handed
        self.tracks['z'] = self.tracks['BCID'].apply(lambda x: x*self.z_width*10000.) #250um z-pitch
        if post_process:
            self.process_digitized()
        if save:
            fname = os.path.splitext(self.tracks_file)[0]
            self.tracks.to_feather(fname + '_digitized.feather')

    def read_tracks(self): #reads relevant colums from the input file
        df = pd.read_feather(self.tracks_file, columns = ['chipx','chipy','chipz','NbEle','tkPDG','eventIonizationEnergy','nHits'])
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
        x_diff = np.sqrt(zs)*self.sigmaT*1e-4*np.random.normal(0,1, len(zs)) #Use random Gaussian smearing to simulate diffusion
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
        if sigma.lower() == 'transverse': #transverse GEM diffusion
            sig = self.sigmaTe
        else:                             #longitudinal GEM diffusion
            sig = self.sigmaLe
        for enum, val in np.ndenumerate(gain_electrons):
            start_ind = np.sum(gain_electrons[:enum[0]])
            end_ind = np.sum(gain_electrons[:enum[0]+1])
            x_post[start_ind:end_ind] = x[enum] + sig*1E-4*np.random.normal(0,1,val) #last term simulated GEM diffusion

    def GEM_gain_and_diffusion(self,track): #Applies gain and diffusion through GEMs
        gain_electrons = np.random.exponential(self.gain, len(track['x'])) #Use random exponential for gain amplification
        gain_electrons = np.asarray(gain_electrons, dtype=int)
        
        x_post = np.ascontiguousarray(np.zeros(np.sum(gain_electrons))) #Use contiguous array for performance
        y_post = np.ascontiguousarray(np.zeros(np.sum(gain_electrons)))
        z_post = np.ascontiguousarray(np.zeros(np.sum(gain_electrons)))
        
        self.generate_gain_points(track['x'], x_post, gain_electrons, sigma='Transverse')
        self.generate_gain_points(track['y'], y_post, gain_electrons, sigma='Transverse')
        self.generate_gain_points(track['z'], z_post, gain_electrons, sigma='Longitudinal')

        return x_post, y_post, z_post

    def digitize_fei4(self,x,y,z):
        xshift = x - x.min()
        yshift = y - y.min()
        zshift = z - z.min()

        #Note: we could find the track center on the chip rather than translating all events to the origin
        #This digitizer was written to create events to feed into a 3D convolutional neural network
        #3DCNNs are translationally invariant, so our approach here is fine
        
        x_bins = np.arange(0, xshift.max()+self.col_width, self.col_width) #Bin with events shifted to origin
        y_bins = np.arange(0, yshift.max()+self.row_width, self.row_width) #Doing so reduces the size of the voxelgrid we digitize over
        z_bins = np.arange(0, zshift.max()+self.z_width,   self.z_width)   #thereby reducing memory consumption and improving speed

        hits = np.array([xshift,yshift,zshift]).T
        pix = np.histogramdd(hits, bins=(x_bins, y_bins, z_bins),
                             range=((0, xshift.max()+self.col_width),
                                    (0, yshift.max()+self.row_width, self.row_width),
                                    (0, zshift.max()+self.z_width, self.z_width)))[0] #Determine 3D pixel hit locations
                
        flat_pix = np.sum(pix, axis=2)

        column, row = np.where(flat_pix > self.threshold) #only one unique x-y pixel hit per event (z is time integrated)
        q = flat_pix[column, row].flatten()
        tot = np.digitize(q, bins=self.pixel_charge_mid, right = True) #Determine the amount of charge in each x,y pixel
        

        bcid = []
        for i in range(0,len(column)):
            bcid_over_thresh = np.where(np.cumsum(pix[column[i],row[i]]) > self.threshold)[0][0] #determine number of cumulative timeslices before charge in a pixel is over threhols
            bcid.append(bcid_over_thresh)                                                        #this gives the BCID which we use as a relative z coordinate

        bcid = np.array(bcid)
        pixel_charge = self.pixel_charge[tot]

        return column, row, bcid, tot, pixel_charge

    def process_digitized(self):

        ### Computations of observables of interest ###

        @jit(nopython=True) #numba for fast computation
        def get_principal_axis(data):
            uu, dd, vv = np.linalg.svd(data-np.array([data[:,0].mean(),data[:,1].mean(),data[:,2].mean()]))
            projection = (data @ vv.T).T[0]
            return projection.max() - projection.min(), vv[0] #returns track length and principal axis vector

        ##Transform carefully to head and tail direction

        def get_angles_and_charge_fractions(data,vec,q):
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
    
        @jit(nopython=True)
        def SDCD(data): #standard deviation of charge distribution
            return np.sqrt(np.diag(data @ data.T).mean())
    
        @jit(nopython=True)
        def wSDCD(wdata,q): #charge weighted standard deviation of charge distribution
            return np.sqrt(np.diag(wdata @ wdata.T).sum()/(q.sum()))

        @jit(nopython=True)
        def compute_y_track_hat(zhat,xhat): #"track coordinate" y unit vector
            return np.cross(zhat,xhat)

        @jit(nopython=True)
        def compute_z_track_hat(xhat,yhat): #"track coordinate" z unit vector
            return np.cross(xhat,yhat)

        @jit(nopython=True)
        def compute_track(data,tuv): #computes track coordinate system; tuv is track unit vector
            return data @ tuv

        def ChargeUnif(data): #std dev of distribution of mean distances between each charge and all other charges
            a = np.linalg.norm(data - data[:,None], axis=-1)
            return np.std([a[i].mean() for i in range(0,len(data))])

        # initial dataframe computations
        
        data = self.tracks
        data['sum_tot'] = data['tot'].apply(lambda x: x.sum()) #total TOT in event
        data['sat_frac'] = data['tot'].apply(lambda x: len(np.where(x == 13)[0])/len(x)) #computes fraction of pixels with TOT = 13
        data['track_charge'] = data['pixel_charge'].apply(lambda x: x.sum()) #sums all hit charges to get total charge of event
        data['track_energy'] = data['pixel_energy'].apply(lambda x: x.sum()) #sums all hit energies to get total energy of event
        data['x_center'] = data['x'] - data['x'].apply(lambda x: x.mean()) #center tracks for track coordinate computations
        data['y_center'] = data['y'] - data['y'].apply(lambda x: x.mean())
        data['z_center'] = data['z'] - data['z'].apply(lambda x: x.mean())

        # lists to fill

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
        ucs = []
        lcs = []
        chargeunif = []
        
        print("Computing e rejection discriminants...")
        zhat = np.array([0,0,1],dtype='float32') #zhat in detector coordinates
        
        for i in tqdm(range(0,len(data))):
            track = np.concatenate([[data['x'][i].T,data['y'][i].T,data['z'][i].T]]).T
            ctrack = np.concatenate([[data['x_center'][i].T,data['y_center'][i].T,data['z_center'][i].T]]).T #centered track
            wtrack = np.concatenate([[data['pixel_charge'][i]*data['x_center'][i].T,data['pixel_charge'][i]*data['y_center'][i].T,data['pixel_charge'][i]*data['z_center'][i].T]]).T # for wSDCD
            q = data['pixel_charge'][i]
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
        data['x_track'] = xtracks
        data['y_track'] = ytracks
        data['z_track'] = ztracks
        data['theta'] = thetas
        data['phi'] = phis
        data['upper_charge_fraction'] = ucfs
        data['upper_charge'] = ucs
        data['lower_charge'] = lcs
        data['LAPA'] = ls
        data['SDCD'] = stds
        data['wSDCD'] = wstds
        data['CylThick'] = (data['y_track']**2+data['z_track']**2).apply(lambda x: x.sum())
        data['ChargeUnif'] = chargeunif
        #data['PrincipalAxis'] = xt_hats #direction along PA
        print('DONE!')        
        return data
        
if __name__ == '__main__':
    np.random.seed(1)
    digi = digitize(post_process = True, save = True) #for interactive session, can also just run with save = True to generate digitized output
