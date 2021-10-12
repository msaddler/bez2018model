import sys
import os
import numpy as np


def rms(x):
    '''
    Returns root mean square amplitude of x (raises ValueError if NaN).
    '''
    out = np.sqrt(np.mean(np.square(x)))
    if np.isnan(out):
        raise ValueError('rms calculation resulted in NaN')
    return out


def get_dBSPL(x, mean_subtract=True):
    '''
    Returns sound pressure level of x in dB re 20e-6 Pa (dB SPL).
    '''
    if mean_subtract:
        x = x - np.mean(x)
    out = 20 * np.log10(rms(x) / 20e-6)
    return out


def set_dBSPL(x, dBSPL, mean_subtract=True):
    '''
    Returns x re-scaled to specified SPL in dB re 20e-6 Pa.
    '''
    if mean_subtract:
        x = x - np.mean(x)
    rms_out = 20e-6 * np.power(10, dBSPL/20)
    return rms_out * x / rms(x)


def combine_signal_and_noise(signal, noise, snr, mean_subtract=True):
    '''
    Adds noise to signal with the specified signal-to-noise ratio (snr).
    If snr is finite, the noise waveform is rescaled and added to the
    signal waveform. If snr is positive infinity, returned waveform is
    equal to the signal waveform. If snr is negative inifinity, returned
    waveform is equal to the noise waveform.
    
    Args
    ----
    signal (np.ndarray): signal waveform
    noise (np.ndarray): noise waveform
    snr (float): signal-to-noise ratio in dB
    mean_subtract (bool): if True, signal and noise are first de-meaned
        (mean_subtract=True is important for accurate snr computation)
    
    Returns
    -------
    signal_and_noise (np.ndarray) signal in noise waveform
    '''
    if mean_subtract:
        signal = signal - np.mean(signal)
        noise = noise - np.mean(noise)        
    if np.isinf(snr) and snr > 0:
        signal_and_noise = signal
    elif np.isinf(snr) and snr < 0:
        signal_and_noise = noise
    else:
        rms_noise_scaling = rms(signal) / (rms(noise) * np.power(10, snr / 20))
        signal_and_noise = signal + rms_noise_scaling * noise
    return signal_and_noise


def power_spectrum(x, fs, rfft=True, dBSPL=True):
    '''
    Helper function for computing power spectrum of sound wave.
    
    Args
    ----
    x (np.ndarray): input waveform (Pa)
    fs (int): sampling rate (Hz)
    rfft (bool): if True, only positive half of power spectrum is returned
    dBSPL (bool): if True, power spectrum has units dB re 20e-6 Pa
    
    Returns
    -------
    freqs (np.ndarray): frequency vector (Hz)
    power_spectrum (np.ndarray): power spectrum (Pa^2 or dB SPL)
    '''
    if rfft:
        # Power is doubled since rfft computes only positive half of spectrum
        power_spectrum = 2 * np.square(np.abs(np.fft.rfft(x) / len(x)))
        freqs = np.fft.rfftfreq(len(x), d=1/fs)
    else:
        power_spectrum = np.square(np.abs(np.fft.fft(x) / len(x)))
        freqs = np.fft.fftfreq(len(x), d=1/fs)
    if dBSPL:
        power_spectrum = 10. * np.log10(power_spectrum / np.square(20e-6)) 
    return freqs, power_spectrum


def complex_tone(f0,
                 fs,
                 dur,
                 harmonic_numbers=[1],
                 frequencies=None,
                 amplitudes=None,
                 phase_mode='sine',
                 offset_start=True,
                 strict_nyquist=True):
    '''
    Function generates a complex harmonic tone with specified relative phase
    and component amplitudes.
    
    Args
    ----
    f0 (float): fundamental frequency (Hz)
    fs (int): sampling rate (Hz)
    dur (float): duration of tone (s)
    harmonic_numbers (list or None): harmonic numbers to include in complex tone (sorted lowest to highest)
    frequencies (list or None): frequencies to include in complex tone (sorted lowest to highest)
    amplitudes (list): RMS amplitudes of individual harmonics (None = equal amplitude harmonics)
    phase_mode (str): specify relative phases (`sch` and `alt` assume contiguous harmonics)
    offset_start (bool): if True, starting phase is offset by np.random.rand()/f0
    strict_nyquist (bool): if True, function will raise ValueError if Nyquist is exceeded;
        if False, frequencies above the Nyquist will be silently ignored
    
    Returns
    -------
    signal (np.ndarray): complex tone
    '''
    # Time vector has step size 1/fs and is of length int(dur*fs)
    t = np.arange(0, dur, 1/fs)[0:int(dur*fs)]
    if offset_start: t = t + (1/f0) * np.random.rand()
    # Create array of frequencies (function requires either harmonic_numbers or frequencies to be specified)
    if frequencies is None:
        assert harmonic_numbers is not None, "cannot specify both `harmonic_numbers` and `frequencies`"
        harmonic_numbers = np.array(harmonic_numbers).reshape([-1])
        frequencies = harmonic_numbers * f0
    else:
        assert harmonic_numbers is None, "cannot specify both `harmonic_numbers` and `frequencies`"
        frequencies = np.array(frequencies).reshape([-1])
    # Set default amplitudes if not provided
    if amplitudes is None:
        amplitudes = 1/len(frequencies) * np.ones_like(frequencies)
    else:
        assert_msg = "provided `amplitudes` must be same length as `frequencies`"
        assert len(amplitudes) == len(frequencies), assert_msg
    # Create array of harmonic phases using phase_mode
    if phase_mode.lower() == 'sine':
        phase_list = np.zeros(len(frequencies))
    elif (phase_mode.lower() == 'rand') or (phase_mode.lower() == 'random'):
        phase_list = 2*np.pi * np.random.rand(len(frequencies))
    elif (phase_mode.lower() == 'sch') or (phase_mode.lower() == 'schroeder'):
        phase_list = np.pi/2 + (np.pi * np.square(frequencies) / len(frequencies))
    elif (phase_mode.lower() == 'cos') or (phase_mode.lower() == 'cosine'):
        phase_list = np.pi/2 * np.ones(len(frequencies))
    elif (phase_mode.lower() == 'alt') or (phase_mode.lower() == 'alternating'):
        phase_list = np.pi/2 * np.ones(len(frequencies))
        phase_list[::2] = 0
    else:
        raise ValueError('Unsupported phase_mode: {}'.format(phase_mode))
    # Build and return the complex tone
    signal = np.zeros_like(t)
    for f, amp, phase in zip(frequencies, amplitudes, phase_list):
        if f > fs/2:
            if strict_nyquist: raise ValueError('Nyquist frequency exceeded')
            else: break
        component = amp * np.sqrt(2) * np.sin(2*np.pi*f*t + phase)
        signal += component
    return signal


def freq2erb(freq):
    '''
    Helper function converts frequency from Hz to ERB-number scale.
    Glasberg & Moore (1990, Hearing Research) equation 4. The ERB-
    number scale can be defined as the number of equivalent
    rectangular bandwidths below the given frequency (units of the
    ERB-number scale are Cams).
    '''
    return 21.4 * np.log10(0.00437 * freq + 1.0)


def erb2freq(erb):
    '''
    Helper function converts frequency from ERB-number scale to Hz.
    Glasberg & Moore (1990, Hearing Research) equation 4. The ERB-
    number scale can be defined as the number of equivalent
    rectangular bandwidths below the given frequency (units of the
    ERB-number scale are Cams).
    '''
    return (1.0/0.00437) * (np.power(10.0, (erb / 21.4)) - 1.0)


def erbspace(freq_min, freq_max, num):
    '''
    Helper function to get array of frequencies linearly spaced on an
    ERB-number scale.
    
    Args
    ----
    freq_min (float): minimum frequency in Hz
    freq_max (float): maximum frequency Hz
    num (int): number of frequencies (length of array)
    
    Returns
    -------
    freqs (np.ndarray): array of ERB-spaced frequencies (lowest to highest) in Hz
    '''
    freqs = np.linspace(freq2erb(freq_min), freq2erb(freq_max), num=num)
    freqs = erb2freq(freqs)
    return freqs
