import numpy as np
import cython_bez2018

fs = 100e3
t = np.arange(0, 2.00, 1/fs)
y = np.sin(2*np.pi*150*t)


for cf in np.linspace(150, 20e3, 50):
    vihc = cython_bez2018.run_ihc(y, fs, cf, species=2)
    print(cf, vihc.shape)
    # d = cython_bez2018.run_synapse(vihc, fs, cf, noiseType=0, implnt=0)
    #
    # for k in d.keys():
    #     print(cf, k, d[k].shape, np.mean(d[k]))
