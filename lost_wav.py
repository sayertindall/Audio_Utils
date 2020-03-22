import numpy as np
import random as rand 
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
import struct
import pyaudio
import time
import wave
from sys import argv

class model:
    
    def __init__(self):
        '''
        Notice that each paramater type has a list with length two. If self.intervals was [(..), (..), (..)]
        then each paramater would have to be length three
        '''
        
        # start
        self.t0 = time.time()
        
        # Each interval is mapped two one group of 'beads' and 'leads' (points)
        self.intervals = [(20,900), (600,1200)] # units in frequency
        
        # -- BEAD PARAMS -- 
        self.bead_count = [3000, 3000]
        self.threshold_beads = .01 # fft amplitude required to contribute to movement
        self.beads = self.make_beads()
        
         # controls how smoothly the marker size reacts to fft data
        self.max_size = [2,7] 
        self.max_freqs = [0 for i in range(len(self.intervals))]
        self.decay = [0.02, 0.006]
        
        # -- LEAD PARAMS --
        self.sensitivity = [0.05, 0.01] # how sensitive they are to fft data which steers the points
        self.leads = {intrv: {'x': rand.uniform(0,1), 'y': rand.uniform(0,1), 
                                 'vx': 0, 'vy': 0, 
                                 'mass': .001} for intrv in self.intervals}
        
         # FFT and angle data
        self.x_freq = np.linspace(0, self.RATE, self.CHUNK)
        self.FFT = np.zeros(self.CHUNK)
        self.mesh = self.make_mesh()

class AudioCapture(model):
    
    # max settings 
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 6
    RATE = 48000

    def __init__(self): 
        super().__init__()
        # self.p = pyaudio.PyAudio()
        self.wf = wave.open(argv[1], 'rb')
        # self.stream = self.p.open(format=self.FORMAT,
        #         channels=self.CHANNELS,
        #         rate=self.RATE,
        #         output=True,
        #         input=True,
        #         frames_per_buffer=self.CHUNK)
        # self.stream.write(data)

    def play(self):
        audio_data = self.wf.readframes(self.CHUNK // 4)
        plt.style.use('dark_background')
        fig, (ax0, ax2) = plt.subplots(2, figsize = (10, 12))
        
        # fft plot and smoothed
        fourier, = ax0.semilogx(self.x_freq, np.zeros(self.CHUNK), '-', c = 'c', lw = 2)
        beads1, = ax2.plot([], [], 'ro', markersize = 1, marker = '.') 
        beads2, = ax2.plot([], [], 'co', markersize = 1, marker = '.') 
        # change color to 'ko' and marker size to 1 to make invisible
        lead1, = ax2.plot([], [], 'ko', markersize = 1, marker = '.') 
        lead2, = ax2.plot([], [], 'ko', markersize = 1, marker = '.')
        
        ax0.set_ylim(-.1, 1)
        ax0.set_xlim(20, self.RATE/2)
        ax0.set_xlabel('frequency')
        ax2.set_xlim(0,1)
        ax2.set_ylim(0,1)
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        while True:
            
            # audio
            prep_data = np.array(struct.unpack(str(self.CHUNK) + 'B', audio_data), dtype='b')[::2]
            audio_data = self.wf.readframes(self.CHUNK // 4)
            self.FFT = abs(fft(prep_data)/(350000)) # arbitrary normalizing constant for FFT 
            # marker_sizes updates points and returns marker sizes after processing FFT data
            marker_sizes = self.update_points(self.FFT) 
            self.border_adjustment()
            
            # unpacking leads and beads data
            Xs, Ys = self.x_y_unpack(self.leads)
            Xb, Yb = self.x_y_unpack(self.beads)
            
            # setting data
            fourier.set_ydata(self.FFT[:self.CHUNK])
            lead1.set_data(Xs[0],Ys[0])
            lead2.set_data(Xs[1],Ys[1])
            beads1.set_data(Xb[0],Yb[0])
            beads2.set_data(Xb[1],Yb[1])
            beads1.set_markersize(marker_sizes[0])
            beads2.set_markersize(marker_sizes[1])
         
            # plot
            fig.canvas.draw()
            fig.canvas.flush_events()
            # plt.show()
            
    # change from dicitonary that need to be unpacked to a straight up np array that can be += in on line
    def update_points(self, FFT):
        
        dt = (time.time()-self.t0)/30
        xx = 0 ; yy = 0 
        for i in range(len(self.intervals)):
            intrvl = self.intervals[i] 
            
            # medium
            d_x = self.beads[intrvl]['x']-self.leads[intrvl]['x']
            d_y = self.beads[intrvl]['y']-self.leads[intrvl]['y']
            dist = np.sqrt(d_x**2 + d_y**2)/5 # /5  is a scale paramater
            self.beads[intrvl]['x'] += (1/dist)*np.cos(dt)*.001
            self.beads[intrvl]['y'] += (1/dist)*np.sin(dt)*.001
            
            mesh_i = self.mesh[i] # [0, (index_0, freq_lo), ..., 2pi, (index_n, freq_hi)] 
            for j in range(len(mesh_i)):
                mesh_ij = mesh_i[j]
                index = mesh_ij[1][0]
                
                # point color and smoothener
                self.max_freqs[i] -= self.decay[i]
                if FFT[index] >= self.max_freqs[i]:
                    self.max_freqs[i] = FFT[index]
                    
                if (FFT[index] >= self.threshold_beads):
                    theta = mesh_ij[0]
                    xx += np.cos(theta + dt) * FFT[index]*self.sensitivity[i] 
                    yy += np.sin(theta + dt) * FFT[index]*self.sensitivity[i] 
                    self.leads[intrvl]['x'] += xx*.001
                    self.leads[intrvl]['y'] += yy*.001
            
        max_freqs = [1+self.max_size[i]*self.max_freqs[i] for i in range(len(self.intervals))]
        return max_freqs
            
                                             
    def x_y_unpack(self, data):
        X, Y = [], []
        for key in data.keys():
            group = data[key]
            X.append(group['x'])
            Y.append(group['y'])
        return X, Y
    
    def border_adjustment(self):
        delta = 0.0001
        for intrvl in self.intervals:
            # leads
            xl = self.leads[intrvl]['x']
            yl = self.leads[intrvl]['y'] 
            if 1-xl < delta:
                self.leads[intrvl]['x'] -= 1
            if 1-yl < delta:
                self.leads[intrvl]['y'] -= 1
            if xl < delta:
                self.leads[intrvl]['x'] += 1
            if yl < delta:
                self.leads[intrvl]['y'] += 1
              
            # beads
            num_beads = self.bead_count[self.intervals.index(intrvl)]
            for i in range(num_beads):
                xb = self.beads[intrvl]['x'][i] 
                yb = self.beads[intrvl]['y'][i] 
                if 1-xb < delta:
                    self.beads[intrvl]['x'][i] -= 1
                if 1-yb < delta:
                    self.beads[intrvl]['y'][i] -= 1
                if xb < delta:
                    self.beads[intrvl]['x'][i] += 1
                if yb < delta:
                    self.beads[intrvl]['y'][i] += 1
                    
    def make_mesh(self):
        # indexing frequencies for ease of subsetting and processing
        freq = np.linspace(0, self.RATE, self.CHUNK)
        freq = [(i, freq[i]) for i in range(len(freq))]
        
        # mapping n intervals of the frequency to [0,2pi]
        mesh = []
        for intrvl in self.intervals:
            f_subset = [i for i in freq if (i[1] > intrvl[0] and i[1] < intrvl[1])]
            angle = np.linspace(0, 2*np.pi, len(f_subset))
            mesh_i = list(zip(angle, f_subset))
            mesh.append(mesh_i)
            
        # for each interval there is a 'mesh': [0, (index, freq_lo), ..., 2pi, (index, freq_hi)] 
        # where freq_lo/hi are the bounds on the interval
        return mesh
    
    def make_beads(self):
        beads = {}
        for i in range(len(self.intervals)):
            count = self.bead_count[i]
            beads.update(
                {self.intervals[i]: {'x': np.array([rand.uniform(0,1) for i in range(count)]),
                    'y': np.array([rand.uniform(0,1) for i in range(count)]),
                    'vx': np.zeros(count),
                    'vy': np.zeros(count)}})
        return beads 
         
    def close(self):
        self.stream.close()
        self.p.terminate()

if __name__ == '__main__':
    audio = AudioCapture()
    audio.play()
    audio.close()

