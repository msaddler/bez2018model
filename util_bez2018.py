import numpy as np
import scipy.signal


def ffGn(N, tdres, Hinput, noiseType, mu, sigma=1):
    """
    Python ffGn implementation based on MATLAB code (ffGn.m); modified from
    https://github.com/mrkrd/cochlea/tree/master/cochlea/zilany2014/util.py
    """
    # Check arguments are valid
    assert (N > 0)
    assert (tdres < 1)
    assert (Hinput >= 0) and (Hinput <= 2)
    
    # Here we change the meaning of `noiseType`, if it's 0, then we
    # return no noise at all.  If necessary, the seed can be set
    # outside by calling np.random.seed()
    if noiseType == 0:
        return np.zeros(N)
    
    # Downsampling No. of points to match with those of Scott Jackson (tau 1e-1)
    resamp = int(np.ceil(1e-1 / tdres))
    nop = N
    N = int(np.ceil(N / resamp) + 1)
    if N < 10: N = 10
    
    # Determine whether fGn or fBn should be produced
    if Hinput <= 1:
        H = Hinput
        fBn = 0
    else:
        H = Hinput - 1
        fBn = 1
    
    # Calculate the fGn
    if H == 0.5:
        # If H=0.5, then fGn is equivalent to white Gaussian noise
        y = np.random.randn(N)
    else:
        # NOTE: ffGn.m uses persistent variables to speed this up
        Nfft = int(2 ** np.ceil(np.log2(2*(N-1))))
        NfftHalf = np.round(Nfft / 2)
        k = np.concatenate( (np.arange(0,NfftHalf), np.arange(NfftHalf,0,-1)) )
        Zmag = 0.5 * ( (k+1)**(2*H) -2*k**(2*H) + np.abs(k-1)**(2*H) )
        Zmag = np.real(np.fft.fft(Zmag))
        assert np.all(Zmag >= 0), 'FFT of the circulant covariance has negative values.'
        Zmag = np.sqrt(Zmag)
        Z = Zmag * (np.random.randn(Nfft) + 1j*np.random.randn(Nfft))
        y = np.real(np.fft.ifft(Z)) * np.sqrt(Nfft)
        y = y[0:N]
        
        # Convert the fGn to fBn, if necessary
        if fBn == 1:
            y = np.cumsum(y)
        
        # Resampling to match with the AN model
        y = scipy.signal.resample(y, resamp*len(y))
        
        # Define standard deviation (ported from BEZ2018/ffGn.m by msaddler)
        if mu < 0.2:
            sigma = 1
        elif mu < 20:
            sigma = 10
        else:
            sigma = mu/2
        y = y*sigma
    
    return y[0:nop]
