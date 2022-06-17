import numpy as np
import pandas as pd
from tqdm import tqdm
import os

class digitize:
    def __init__(self, col_width = .025, row_width = .0050, z_width = .0250, threshold = 3137, gain = 15000, sigmaT = 134.8, sigmaL = 128.2, sigmaTe = 143, sigmaLe = 97, pixel_charge = np.array([3237, 3253.2, 3681.4, 4936.4, 7224.8, 8972.2, 12143.7, 16204, 20159, 24351, 28696, 33368, 38931, 52141]),tracks_file = 'test_to_digitize.feather', save = True):
        
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
            #except:
            #    column.append(np.nan)
            #    row.append(np.nan)
            #    BCID.append(np.nan)
            #    tot.append(np.nan)
            #    q.append(np.nan)
        self.tracks['column'] = column
        self.tracks['row'] = row
        self.tracks['BCID'] = BCID
        self.tracks['tot'] = tot
        self.tracks['pixel_charge'] = q
        if save:
            fname = os.path.splitext(self.tracks_file)[0]
            self.tracks.to_feather(fname + '_digitized.feather')

    def read_tracks(self):
        df = pd.read_feather(self.tracks_file, columns = ['chipx','chipy','chipz','NbEle','tkPDG','eventIonizationEnergy','nHits'])
        return df

    def apply_diffusion(self, track):
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

    def generate_gain_points(self, x, x_post, gain_electrons, sigma = 'Transverse'):
        if sigma.lower() == 'transverse':
            sig = self.sigmaTe
        else:
            sig = self.sigmaLe
        for enum, val in np.ndenumerate(gain_electrons):
            start_ind = np.sum(gain_electrons[:enum[0]])
            end_ind = np.sum(gain_electrons[:enum[0]+1])
            x_post[start_ind:end_ind] = x[enum] + sig*1E-4*np.random.normal(0,1,val)

    def GEM_gain_and_diffusion(self,track):
        gain_electrons = np.random.exponential(self.gain, len(track['x']))
        gain_electrons = np.asarray(gain_electrons, dtype=int)
        
        x_post = np.ascontiguousarray(np.zeros(np.sum(gain_electrons)))
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

        x_bins = np.arange(0, xshift.max()+self.col_width, self.col_width)
        y_bins = np.arange(0, yshift.max()+self.row_width, self.row_width)
        z_bins = np.arange(0, zshift.max()+self.z_width,   self.z_width)

        hits = np.array([xshift,yshift,zshift]).T
        pix = np.histogramdd(hits, bins=(x_bins, y_bins, z_bins),
                             range=((0, xshift.max()+self.col_width),
                                    (0, yshift.max()+self.row_width, self.row_width),
                                    (0, zshift.max()+self.z_width, self.z_width)))[0]
                
        flat_pix = np.sum(pix, axis=2)

        column, row = np.where(flat_pix > self.threshold)
        q = flat_pix[column, row].flatten()
        tot = np.digitize(q, bins=self.pixel_charge_mid, right = True)
        

        bcid = []
        for i in range(0,len(column)):
            bcid_over_thresh = np.where(np.cumsum(pix[column[i],row[i]]) > self.threshold)[0][0]
            bcid.append(bcid_over_thresh)

        bcid = np.array(bcid)
        #bcid = bcid - bcid.min()
        #tot = tot -1
        pixel_charge = self.pixel_charge[tot]

        return column, row, bcid, tot, pixel_charge
        
if __name__ == '__main__':
    np.random.seed(1)
    a = digitize()
