import sys
import os
import numpy as np
import matplotlib.pyplot
import matplotlib.ticker


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

def format_axes(ax,
                str_title=None,
                str_xlabel=None,
                str_ylabel=None,
                fontsize_title=12,
                fontsize_labels=12,
                fontsize_ticks=12,
                fontweight_title=None,
                fontweight_labels=None,
                xscale='linear',
                yscale='linear',
                xlimits=None,
                ylimits=None,
                xticks=None,
                yticks=None,
                xticks_minor=None,
                yticks_minor=None,
                xticklabels=None,
                yticklabels=None,
                spines_to_hide=[],
                major_tick_params_kwargs_update={},
                minor_tick_params_kwargs_update={}):
    '''
    Helper function for setting axes-related formatting parameters.
    '''
    ax.set_title(str_title, fontsize=fontsize_title, fontweight=fontweight_title)
    ax.set_xlabel(str_xlabel, fontsize=fontsize_labels, fontweight=fontweight_labels)
    ax.set_ylabel(str_ylabel, fontsize=fontsize_labels, fontweight=fontweight_labels)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)
    
    if xticks_minor is not None:
        ax.set_xticks(xticks_minor, minor=True)
    if yticks_minor is not None:
        ax.set_yticks(yticks_minor, minor=True)
    if xticks is not None:
        ax.set_xticks(xticks, minor=False)
    if yticks is not None:
        ax.set_yticks(yticks, minor=False)
    if xticklabels is not None:
        ax.set_xticklabels([], minor=True)
        ax.set_xticklabels(xticklabels, minor=False)
    if yticklabels is not None:
        ax.set_yticklabels([], minor=True)
        ax.set_yticklabels(yticklabels, minor=False)
    
    major_tick_params_kwargs = {
        'axis': 'both',
        'which': 'major',
        'labelsize': fontsize_ticks,
        'length': fontsize_ticks/2,
        'direction': 'out',
    }
    major_tick_params_kwargs.update(major_tick_params_kwargs_update)
    ax.tick_params(**major_tick_params_kwargs)
    
    minor_tick_params_kwargs = {
        'axis': 'both',
        'which': 'minor',
        'labelsize': fontsize_ticks,
        'length': fontsize_ticks/4,
        'direction': 'out',
    }
    minor_tick_params_kwargs.update(minor_tick_params_kwargs_update)
    ax.tick_params(**minor_tick_params_kwargs)
    
    for spine_key in spines_to_hide:
        ax.spines[spine_key].set_visible(False)
    
    return ax


def make_line_plot(ax,
                   x,
                   y,
                   legend_on=False,
                   kwargs_plot={},
                   kwargs_legend={},
                   **kwargs_format_axes):
    '''
    Helper function for basic line plot with optional legend.
    '''
    kwargs_plot_tmp = {
        'marker': '',
        'ls': '-',
        'color': [0, 0, 0],
        'lw': 1,
    }
    kwargs_plot_tmp.update(kwargs_plot)
    ax.plot(x, y, **kwargs_plot_tmp)
    ax = format_axes(ax, **kwargs_format_axes)
    if legend_on:
        kwargs_legend_tmp = {
            'loc': 'lower right',
            'frameon': False,
            'handlelength': 1.0,
            'markerscale': 1.0,
            'fontsize': 12,
        }
        kwargs_legend_tmp.update(kwargs_legend)
        ax.legend(**kwargs_legend_tmp)
    return ax


def make_nervegram_plot(ax,
                        nervegram,
                        sr=20000,
                        cfs=None,
                        cmap='gray',
                        cbar_on=False,
                        fontsize_labels=12,
                        fontsize_ticks=12,
                        fontweight_labels=None,
                        nxticks=6,
                        nyticks=5,
                        tmin=None,
                        tmax=None,
                        treset=True,
                        vmin=None,
                        vmax=None,
                        vticks=None,
                        str_clabel=None,
                        **kwargs_format_axes):
    '''
    Helper function for visualizing auditory nervegram (or similar) representation.
    '''
    # Trim nervegram if tmin and tmax are specified
    nervegram = np.squeeze(nervegram)
    assert len(nervegram.shape) == 2, "nervegram must be freq-by-time array"
    t = np.arange(0, nervegram.shape[1]) / sr
    if (tmin is not None) and (tmax is not None):
        t_IDX = np.logical_and(t >= tmin, t < tmax)
        t = t[t_IDX]
        nervegram = nervegram[:, t_IDX]
    if treset:
        t = t - t[0]
    # Setup time and frequency ticks and labels
    time_idx = np.linspace(0, t.shape[0]-1, nxticks, dtype=int)
    time_labels = ['{:.0f}'.format(1e3 * t[itr0]) for itr0 in time_idx]
    if cfs is None:
        cfs = np.arange(0, nervegram.shape[0])
    else:
        cfs = np.array(cfs)
        assert cfs.shape[0] == nervegram.shape[0], "cfs.shape[0] must match nervegram.shape[0]"
    freq_idx = np.linspace(0, cfs.shape[0]-1, nyticks, dtype=int)
    freq_labels = ['{:.0f}'.format(cfs[itr0]) for itr0 in freq_idx]
    # Display nervegram image
    im_nervegram = ax.imshow(nervegram,
                             origin='lower',
                             aspect='auto',
                             extent=[0, nervegram.shape[1], 0, nervegram.shape[0]],
                             cmap=cmap,
                             vmin=vmin,
                             vmax=vmax)
    # Add colorbar if `cbar_on == True`
    if cbar_on:
        cbar = matplotlib.pyplot.colorbar(im_nervegram, ax=ax, pad=0.02)
        cbar.ax.set_ylabel(str_clabel, fontsize=fontsize_labels, fontweight=fontweight_labels)
        if vticks is not None:
            cbar.set_ticks(vticks)
        else:
            cbar.ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nyticks, integer=True))
        cbar.ax.tick_params(direction='out',
                            axis='both',
                            which='both',
                            labelsize=fontsize_ticks,
                            length=fontsize_ticks/2)
        cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%03d'))
    # Format axes
    ax = format_axes(ax,
                     xticks=time_idx,
                     yticks=freq_idx,
                     xticklabels=time_labels,
                     yticklabels=freq_labels,
                     fontsize_labels=fontsize_labels,
                     fontsize_ticks=fontsize_ticks,
                     fontweight_labels=fontweight_labels,
                     **kwargs_format_axes)
    return ax


def make_stimulus_summary_plot(ax_arr,
                               ax_idx_waveform=None,
                               ax_idx_spectrum=None,
                               ax_idx_nervegram=None,
                               ax_idx_excitation=None,
                               waveform=None,
                               nervegram=None,
                               sr_waveform=None,
                               sr_nervegram=None,
                               cfs=None,
                               tmin=None,
                               tmax=None,
                               treset=True,
                               vmin=None,
                               vmax=None,
                               n_anf=None,
                               erb_freq_axis=True,
                               spines_to_hide_waveform=[],
                               spines_to_hide_spectrum=[],
                               spines_to_hide_excitation=[],
                               nxticks=6,
                               nyticks=6,
                               kwargs_plot={},
                               limits_buffer=0.1,
                               ax_arr_clear_leftover=True,
                               **kwargs_format_axes):
    '''
    Helper function for generating waveform, power spectrum, nervegram, and excitation pattern
    plots to summarize a stimulus.
    '''
    # Axes are tracked in flattened array
    ax_arr = np.array([ax_arr]).reshape([-1])
    assert len(ax_arr.shape) == 1
    ax_idx_list = []
    
    # Plot stimulus waveform
    if ax_idx_waveform is not None:
        ax_idx_list.append(ax_idx_waveform)
        y_wav = np.squeeze(waveform)
        assert len(y_wav.shape) == 1, "waveform must be 1D array"
        x_wav = np.arange(0, y_wav.shape[0]) / sr_waveform
        if (tmin is not None) and (tmax is not None):
            IDX = np.logical_and(x_wav >= tmin, x_wav < tmax)
            x_wav = x_wav[IDX]
            y_wav = y_wav[IDX]
        if treset:
            x_wav = x_wav - x_wav[0]
        xlimits_wav = [x_wav[0], x_wav[-1]]
        ylimits_wav = [np.max(np.abs(y_wav)), -np.max(np.abs(y_wav))]
        ylimits_wav = np.array(ylimits_wav) * (1 + limits_buffer)
        make_line_plot(ax_arr[ax_idx_waveform],
                       x_wav,
                       y_wav,
                       legend_on=False,
                       kwargs_plot=kwargs_plot,
                       kwargs_legend={},
                       xlimits=xlimits_wav,
                       ylimits=ylimits_wav,
                       xticks=[],
                       yticks=[],
                       xticklabels=[],
                       yticklabels=[],
                       spines_to_hide=spines_to_hide_waveform,
                       **kwargs_format_axes)
    
    # Plot stimulus power spectrum
    if ax_idx_spectrum is not None:
        ax_idx_list.append(ax_idx_spectrum)
        fxx, pxx = power_spectrum(waveform, sr_waveform)
        if cfs is not None:
            IDX = np.logical_and(fxx >= np.min(cfs), fxx <= np.max(cfs))
            pxx = pxx[IDX]
            fxx = fxx[IDX]
        xlimits_pxx = [np.max(pxx) * (1 + limits_buffer), 0] # Reverses x-axis                                                                                                     
        xlimits_pxx = np.ceil(np.array(xlimits_pxx) * 5) / 5
        if erb_freq_axis:
            fxx = freq2erb(fxx)
            xlimits_buffer_pxx = limits_buffer * np.max(pxx)
            ylimits_fxx = [np.min(fxx), np.max(fxx)]
            yticks = np.linspace(ylimits_fxx[0], ylimits_fxx[-1], nyticks)
            yticklabels = ['{:.0f}'.format(yt) for yt in erb2freq(yticks)]
        else:
            ylimits_fxx = [np.min(fxx), np.max(fxx)]
            yticks = np.linspace(ylimits_fxx[0], ylimits_fxx[-1], nyticks)
            yticklabels = ['{:.0f}'.format(yt) for yt in yticks]
        make_line_plot(ax_arr[ax_idx_spectrum],
                       pxx,
                       fxx,
                       legend_on=False,
                       kwargs_plot=kwargs_plot,
                       str_xlabel='Power\n(dB SPL)',
                       str_ylabel='Frequency (Hz)',
                       xlimits=xlimits_pxx,
                       ylimits=ylimits_fxx,
                       xticks=xlimits_pxx,
                       yticks=yticks,
                       xticklabels=xlimits_pxx.astype(int),
                       yticklabels=yticklabels,
                       spines_to_hide=spines_to_hide_spectrum,
                       **kwargs_format_axes)
    
    # Plot stimulus nervegram
    if ax_idx_nervegram is not None:
        ax_idx_list.append(ax_idx_nervegram)
        if ax_idx_spectrum is not None:
            nervegram_nxticks = nxticks
            nervegram_nyticks = 0
            nervegram_str_xlabel = 'Time\n(ms)'
            nervegram_str_ylabel = None
        else:
            nervegram_nxticks = nxticks
            nervegram_nyticks = nyticks
            nervegram_str_xlabel = 'Time (ms)'
            nervegram_str_ylabel = 'Characteristic frequency (Hz)'
        make_nervegram_plot(ax_arr[ax_idx_nervegram],
                            nervegram,
                            sr=sr_nervegram,
                            cfs=cfs,
                            nxticks=nervegram_nxticks,
                            nyticks=nervegram_nyticks,
                            tmin=tmin,
                            tmax=tmax,
                            treset=treset,
                            vmin=vmin,
                            vmax=vmax,
                            str_xlabel=nervegram_str_xlabel,
                            str_ylabel=nervegram_str_ylabel)
    
    # Plot stimulus excitation pattern
    if ax_idx_excitation is not None:
        ax_idx_list.append(ax_idx_excitation)
        if np.all(np.mod(nervegram, 1) == 0):
            # Compute mean firing rate from spike counts if all values are integers
            x_exc = np.sum(nervegram, axis=1) / (nervegram.shape[1] / sr_nervegram)
            if n_anf is not None:
                # If a number of ANFs is specified, divide firing rate by n_anf
                x_exc = x_exc / n_anf
        else:
            # Otherwise, compute mean firing rates from instantaneous firing rates
            x_exc = np.mean(nervegram, axis=1)
        xlimits_exc = [0, np.max(x_exc) * (1 + limits_buffer)]
        xlimits_exc = np.ceil(np.array(xlimits_exc)/10) * 10
        y_exc = np.arange(0, nervegram.shape[0])
        ylimits_exc = [np.min(y_exc), np.max(y_exc)]
        make_line_plot(ax_arr[ax_idx_excitation],
                       x_exc,
                       y_exc,
                       legend_on=False,
                       kwargs_plot=kwargs_plot,
                       str_xlabel='Excitation\n(spikes/s)',
                       xlimits=xlimits_exc,
                       ylimits=ylimits_exc,
                       xticks=xlimits_exc,
                       yticks=[],
                       xticklabels=xlimits_exc.astype(int),
                       yticklabels=[],
                       spines_to_hide=spines_to_hide_excitation,
                       **kwargs_format_axes)
    
    # Clear unused axes in ax_arr
    if ax_arr_clear_leftover:
        for ax_idx in range(ax_arr.shape[0]):
            if ax_idx not in ax_idx_list:
                ax_arr[ax_idx].axis('off')
    return ax_arr
