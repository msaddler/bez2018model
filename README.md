## Python wrapper around the Bruce, Erfani, and Zilany (2018) auditory nerve model

The [source code](https://www.ece.mcmaster.ca/~ibruce/zbcANmodel/zbcANmodel.htm) for the Bruce et al. (2018) auditory nerve model was modified to interface with Python and allow easy manipulation of parameters such as cochlear filter bandwidths and inner hair cell filter cutoffs. This wrapper was developed for the experiments in our [paper](https://doi.org/10.1038/s41467-021-27366-6):

    "Deep neural network models reveal interplay of peripheral coding and stimulus statistics in pitch perception".
    Mark R. Saddler, Ray Gonzalez, Josh H. McDermott (Nature Communications, 2021).

Contact: Mark R. Saddler (msaddler@mit.edu)


### Installation

```
(0) Required Python packages:

Package              Version
-------------------- ----------------------
Cython               0.29.24
h5py                 2.9.0
numpy                1.16.3
scipy                1.4.1

(1) clone bez2018model repository
(2) cd bez2018model
(3) python setup.py build_ext --inplace
```


### Example usage

```
import bez2018model
signal <- audio waveform with shape [timesteps] or [timesteps, channels]
signal_fs <- sampling rate of signal in Hz
kwargs <- keyword arguments for bez2018model.nervegram specifing auditory nerve model parameters
nervegram_output_dict = bez2018model.nervegram(signal, signal_fs, **kwargs)
```


### README from the Bruce et al. (2018) source code

This is the BEZ2018 version of the code for auditory periphery model from
the Carney, Bruce and Zilany labs.

This release implements the version of the model described in:

    Bruce, I.C., Erfani, Y., and Zilany, M.S.A. (2018). "A Phenomenological
    model of the synapse between the inner hair cell and auditory nerve: 
    Implications of limited neurotransmitter release sites," to appear in
    Hearing Research. (Special Issue on "Computational Models in Hearing".)

Please cite this paper if you publish any research results obtained with
this code or any modified versions of this code.
