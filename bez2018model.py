import cython_bez2018 # Package must be installed in-place: `python setup.py build_ext --inplace`
import numpy as np
import scipy.signal


def get_ERB_cf_list(num_cfs, min_cf=125., max_cf=10e3):
    '''
    Helper function to get array of num_cfs ERB-scaled CFs between min_cf and max_cf.
    
    Args
    ----
    num_cfs (int): number of CFs sets length of output list
    min_cf (float): lowest CF in Hz
    max_cf (float): highest CF in Hz
    
    Returns
    -------
    cf_list (list): list of CFs in Hz (lowest to highest)
    '''
    E_start = 21.4 * np.log10(0.00437 * min_cf + 1.0)
    E_end = 21.4 * np.log10(0.00437 * max_cf + 1.0)
    cf_list = np.linspace(E_start, E_end, num = num_cfs)
    cf_list = (1.0/0.00437) * (10.0 ** (cf_list / 21.4) - 1.0)
    return list(cf_list)


def apply_lowpass_filter(x, fs, axis=1, order=6, cutoff=0):
    '''
    Helper function to apply lowpass butterworth filter to an array with filtfilt.
    
    Args
    ----
    x (np array): array to be filtered
    fs (int): sampling rate of x in Hz
    axis (int): axis along which to apply filter
    order (int): filter order (passed to scipy.signal.butter); NOTE filter is applied twice
    cutoff (float): butterworth filter -3 dB cutoff frequency; NOTE filter is applied twice
    
    Returns
    -------
    x (np array): lowpass filtered x
    '''
    if cutoff > 0:
        b, a = scipy.signal.butter(order, cutoff/(fs/2), btype='lowpass')
        x = scipy.signal.filtfilt(b, a, x, axis=axis)
    else: print('No lowpass filter applied (cutoff={}Hz)'.format(cutoff))
    return x


def nervegram_meanrates(signal, signal_fs, meanrates_params={}, ANmodel_params={},
                        lpfilter_params={}, random_seed=None):
    '''
    Main function for generating an auditory nervegram using the meanrates output
    from the BEZ2018 ANmodel (analytic estimate of instantaneous firing rate).
    
    Args
    ----
    signal (np array): input pressure waveform must be 1-dimensional array (units Pa)
    signal_fs (int): sampling rate of signal
    meanrates_params (dict): parameters for formatting output nervegram
    ANmodel_params (dict): parameters for running BEZ2018 ANmodel
    lpfilter_params (dict): parameters for applying a lowpass filter to nervegram
    random_seed (int or None): if not None, used to set np.random.seed
    
    Returns
    -------
    output_dict (dict): contains meanrates nervegram, stimulus, and all parameters
    '''
    # ======================== PARSE PARAMETERS ======================== #
    
    # If specified, set random seed (eliminates stochasticity in ANmodel noise)
    if not (random_seed == None): np.random.seed(random_seed)
    
    # BEZ2018 ANmodel requires 1 dimensional arrays with dtype np.float64
    signal = np.squeeze(signal).astype(np.float64)
    assert len(signal.shape) == 1, "signal must be a 1-dimensional array"
    signal_dur = signal.shape[0] / signal_fs
    
    # Nervegram duration / sampling rate / segment clipping parameters
    meanrates_dur = meanrates_params.get('dur', signal_dur) # nervegram duration (s)
    meanrates_fs = meanrates_params.get('fs', 10e3) # nervegram sampling rate (Hz)
    buffer_start_dur = meanrates_params.get('buffer_start_dur', 0.070) # ignore period at start of nervegram (s)
    buffer_end_dur = meanrates_params.get('buffer_end_dur', 0.010) # ignore period at end of nervegram (s)
    
    # BEZ2018 ANmodel and sound presentation level parameters
    pin_fs = ANmodel_params.get('pin_fs', 100e3) # sampling rate for ANmodel input (Hz)
    pin_dBSPL_flag = ANmodel_params.get('pin_dBSPL_flag', 0) # if 1, pin will be re-scaled to specified SPL
    pin_dBSPL = ANmodel_params.get('pin_dBSPL', 65.0) # ANmodel stimulus presentation level (dB SPL)
    species = ANmodel_params.get('species', 2) # 1=cat, 2=human (Shera et al. 2002), 3=human (Glasberg&Moore 1990)
    bandwidth_scale_factor = ANmodel_params.get('bandwidth_scale_factor', 1.0) # Cochlear filter BW scaling factor
    cohc = ANmodel_params.get('cohc', 1.0) # OHC scaling factor: 1=normal OHC function, 0=complete OHC dysfunction
    cihc = ANmodel_params.get('cihc', 1.0) # IHC scaling factor: 1=normal IHC function, 0=complete IHC dysfunction
    noiseType = ANmodel_params.get('noiseType', 1) # set to 0 for noiseless and 1 for variable fGn
    implnt = ANmodel_params.get('implnt', 0) # set to 0 for "approx" and 1 for "actual" power-law implementation
    tabs = ANmodel_params.get('tabs', 0.6e-3) # absolute refractory period (s)
    trel = ANmodel_params.get('trel', 0.6e-3) # baseline mean relative refractory period (s)
    
    # Specify spontaneous firing rates and characteristic frequencies
    spont_list = ANmodel_params.get('spont_list', 70.0) # spontaneous firing rate(s) (spikes/s)
    spont_list = np.array(spont_list).reshape([-1]).tolist() # spont_list can be one value or a list of values
    if 'cf_list' in ANmodel_params.keys(): # list of characteristic frequencies (Hz)
        cf_list = ANmodel_params['cf_list']
    else: # if list is not provided, build list of ERB-spaced characteristic frequencies (Hz)
        num_cfs = ANmodel_params.get('num_cfs', 50)
        min_cf = ANmodel_params.get('min_cf', 125.)
        max_cf = ANmodel_params.get('max_cf', 10e3)
        cf_list = get_ERB_cf_list(num_cfs, min_cf=min_cf, max_cf=max_cf)
    
    # ======================== PREPARE INPUT SIGNAL ======================== #
    
    # Resample the input signal to pin_fs (at least 100kHz) for ANmodel
    pin = scipy.signal.resample_poly(signal, int(pin_fs), int(signal_fs))
    
    # If pin_dBSPL_flag, scale pin to desired dB SPL (otherwise compute dB SPL)
    if pin_dBSPL_flag:
        desired_rms = 2e-5 * np.power(10, pin_dBSPL / 20)
        pin = pin - np.mean(mean)
        pin = desired_rms * (pin / np.sqrt(np.mean(np.square(pin))))
    else:
        pin_dBSPL = 20 * np.log10(np.sqrt(np.mean(np.square(pin))) / 2e-5)
    
    # ======================== RUN AUDITORY NERVE MODEL ======================== #
    
    # Initialize output (downsample pin to get the correct time dimension length)
    decimated_pin = scipy.signal.resample_poly(pin, int(meanrates_fs), int(pin_fs))
    meanrates = np.zeros((len(cf_list), decimated_pin.shape[0], len(spont_list)))

    # Iterate over all CFs and spont rates (only synapse model uses spont rate)
    for cf_idx, cf in enumerate(cf_list):
        ###### Run IHC model ######
        vihc = cython_bez2018.run_ihc(pin, pin_fs, cf,
                                      species=species,
                                      bandwidth_scale_factor=bandwidth_scale_factor,
                                      cohc=cohc,
                                      cihc=cihc)                
        for spont_idx, spont in enumerate(spont_list):
            ###### Run IHC-ANF synapse model ######
            synapse_out = cython_bez2018.run_synapse(vihc, pin_fs, cf,
                                                     noiseType=noiseType,
                                                     implnt=implnt,
                                                     spont=spont,
                                                     tabs=tabs,
                                                     trel=trel)
            # Downsample meanrates to meanrates_fs
            meanrate = scipy.signal.resample_poly(synapse_out['meanrate'], int(meanrates_fs), int(pin_fs))
            meanrate[meanrate < 0] = 0 # Half-wave rectify to remove negative artifacts from downsampling
            meanrates[cf_idx, :, spont_idx] = meanrate
    
    # ======================== APPLY MANIPULATIONS ======================== #
    
    # Apply lowpass filter to nervegram (if lpfilter_params is specified)
    if lpfilter_params:
        meanrates = apply_lowpass_filter(meanrates, meanrates_fs, axis=1, **lpfilter_params)
    
    # Randomly clip a segment of duration meanrates_dur from the larger nervegram
    (clip_start_meanrates, clip_end_meanrates) = (0, meanrates.shape[1])
    (clip_start_pin, clip_end_pin) = (0, pin.shape[0])
    (signal_clip_start, signal_clip_end) = (0, signal.shape[0])
    if meanrates_dur < signal_dur:
        buffer_start_idx = np.ceil(buffer_start_dur*meanrates_fs)
        buffer_end_idx = meanrates.shape[1] - np.floor(buffer_end_dur*meanrates_fs)
        clip_start_meanrates = np.random.randint(buffer_start_idx,
                                                 high=buffer_end_idx-meanrates_dur*meanrates_fs)
        clip_end_meanrates = clip_start_meanrates + int(np.ceil(meanrates_dur*meanrates_fs))
        assert clip_end_meanrates <= buffer_end_idx, "clip_end_meanrates out of buffered range"
        meanrates = meanrates[:, clip_start_meanrates:clip_end_meanrates]
        # Clip analogous segment of pin (stimulus provided to ANmodel)
        clip_start_pin = int(clip_start_meanrates * pin_fs / meanrates_fs)
        clip_end_pin = int(clip_end_meanrates * pin_fs / meanrates_fs)
        pin = pin[clip_start_pin:clip_end_pin]
        # Clip analogous segment of signal (input stimulus)
        clip_start_signal = int(clip_start_meanrates * signal_fs / meanrates_fs)
        clip_end_signal = int(clip_end_meanrates * signal_fs / meanrates_fs)
        signal = signal[clip_start_signal:clip_end_signal]
    
    # ======================== ORGANIZE output_dict ======================== #
    
    output_dict = {
        'signal': signal.astype(np.float32),
        'signal_fs': signal_fs,
        'pin': pin.astype(np.float32),
        'pin_fs': pin_fs,
        'meanrates': meanrates.astype(np.float32),
        'meanrates_fs': meanrates_fs,
        'cf_list': np.array(cf_list).astype(np.float32),
        'spont_list': np.array(spont_list).astype(np.float32),
        'meanrates_dur': meanrates_dur,
        'buffer_start_dur': buffer_start_dur,
        'buffer_end_dur': buffer_end_dur,
        'pin_dBSPL_flag': pin_dBSPL_flag,
        'pin_dBSPL': pin_dBSPL,
        'species': species,
        'bandwidth_scale_factor': bandwidth_scale_factor,
        'cohc': cohc,
        'cihc': cihc,
        'noiseType': noiseType,
        'implnt': implnt,
        'tabs': tabs,
        'trel': trel,
    }
    for key in lpfilter_params:
        output_dict['lpfilter_' + key] = lpfilter_params[key]
    
    return output_dict
