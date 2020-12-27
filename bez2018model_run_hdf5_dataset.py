import sys
import os
import bez2018model
import numpy as np
import h5py
import glob
import re
import time
import copy
import collections
try:
    import dask.array
except ImportError as e:
    print(e)
    pass


def recursive_dict_merge(dict1, dict2):
    '''
    Returns a new dictionary by merging two dictionaries recursively.
    This function is useful for minimally updating dict1 with dict2.
    '''
    result = copy.deepcopy(dict1)
    for key, value in dict2.items():
        if isinstance(value, collections.Mapping):
            result[key] = recursive_dict_merge(result.get(key, {}), value)
        else:
            result[key] = copy.deepcopy(dict2[key])
    return result


def write_example_to_hdf5(hdf5_f,
                          data_dict,
                          idx,
                          data_key_pair_list=[]):
    '''
    Write individual example to open hdf5 file.

    Args
    ----
    hdf5_f (h5py.File object): open and writeable hdf5 file object
    data_dict (dict): dict containing auditory nerve model output
    idx (int): specifies row of hdf5 file to write to
    data_key_pair_list (list): list of tuples (hdf5_key, data_key)
    '''
    for (hdf5_key, data_key) in data_key_pair_list:
        hdf5_f[hdf5_key][idx] = np.squeeze(np.array(data_dict[data_key]))


def initialize_hdf5_file(hdf5_filename,
                         N,
                         data_dict,
                         file_mode='w',
                         data_key_pair_list=[],
                         config_key_pair_list=[],
                         dtype=np.float32,
                         cast_data=False,
                         cast_config=False,
                         compression=None,
                         compression_opts=None):
    '''
    Create a new hdf5 file and populate config parameters.

    Args
    ----
    hdf5_filename (str): filename for new hdf5 file
    N (int): number of examples that will be written to file (number of rows for datasets)
    data_dict (dict): dict containing auditory nerve model output and all metadata
    file_mode (str): 'w' = Create file, truncate if exists; 'w-' = Create file, fail if exists
    data_key_pair_list (list): list of tuples (hdf5_key, data_key) for datasets with N rows
    config_key_pair_list (list): list of tuples (hdf5_key, config_key) for config datasets
    dtype (np.dtype object): datatype for casting data and/or config values
    cast_data (bool): if True, fields corresponding to data_key_pair_list will be cast to dtype
    cast_config (bool): if True, fields corresponding to config_key_pair_list will be cast to dtype
    compression (str): if not None, specifies compression filter for  hdf5 datasets (example: 'gzip')
    compression_opts (int): options for compression filter (for 'gzip', sets compression level: 0 to 9)
    '''
    # Initialize hdf5 file
    f = h5py.File(hdf5_filename, file_mode)
    kwargs_create_dataset = {}
    if compression is not None:
        kwargs_create_dataset['compression'] = compression
    if compression_opts is not None:
        kwargs_create_dataset['compression_opts'] = compression_opts
    # Create the main output datasets
    for (hdf5_key, data_key) in data_key_pair_list:
        data_key_value = np.squeeze(np.array(data_dict[data_key]))
        if cast_data:
            data_key_value = data_key_value.astype(dtype)
        if ('vlen' in hdf5_key) or ('sparse' in hdf5_key):
            data_key_dtype = h5py.special_dtype(vlen=data_key_value.dtype)
            data_key_shape = [N] + list(data_key_value.shape)[:-1]
        else:
            data_key_dtype = data_key_value.dtype
            data_key_shape = [N] + list(data_key_value.shape)
        f.create_dataset(hdf5_key,
                         data_key_shape,
                         dtype=data_key_dtype,
                         **kwargs_create_dataset)
    # Create and populate the config datasets
    for (hdf5_key, config_key) in config_key_pair_list:
        config_key_value = np.squeeze(np.array(data_dict[config_key]))
        if cast_config:
            config_key_value = config_key_value.astype(dtype)
        config_key_shape = [1] + list(config_key_value.shape)
        f.create_dataset(hdf5_key,
                         config_key_shape,
                         dtype=config_key_value.dtype,
                         data=config_key_value,
                         **kwargs_create_dataset)
    # Close the initialized hdf5 file
    f.close()


def check_continuation(hdf5_filename,
                       check_key='/pin_dBSPL',
                       check_key_fill_value=0,
                       repeat_buffer=1):
    '''
    This function checks if the output dataset already exists and should be continued
    from the last populated row rather than restarted.

    Args
    ----
    hdf5_filename (str): filename for hdf5 dataset to check
    check_key (str): key in hdf5 file used to check for continuation (should be 1-dimensional dataset)
    check_key_fill_value (int or float): function will check for rows where check_key is equal to this value
    repeat_buffer (int): if continuing existing file, number of rows should be re-processed

    Returns
    -------
    continuation_flag (bool): True if hdf5 file exists and can be continued
    start_idx (int or None): row of hdf5 dataset at which to begin continuation
    '''
    continuation_flag = False
    start_idx = 0
    if os.path.isfile(hdf5_filename):
        f = h5py.File(hdf5_filename, 'r')
        if check_key in f:
            candidate_idxs = np.reshape(np.argwhere(f[check_key][:] == check_key_fill_value), [-1])
            continuation_flag = True
            if len(candidate_idxs > 0):
                start_idx = np.max([0, np.min(candidate_idxs)-repeat_buffer])
            else:
                start_idx = None
        else:
            print('>>> [WARNING] check_key not found in hdf5 file; hdf5 dataset will be restarted')
        f.close()
    return continuation_flag, start_idx


def get_default_data_key_pair_list(data_dict,
                                   hdf5_key_prefix='',
                                   data_keys=['signal',
                                              'pin_dBSPL',
                                              'nervegram_vihcs',
                                              'nervegram_meanrates',
                                              'nervegram_spike_times',
                                              'nervegram_spike_tensor',
                                              'nervegram_spike_tensor_dense',
                                              'nervegram_spike_tensor_sparse*']):
    '''
    Helper function to get default data_key_pair_list from data_dict.

    Args
    ----
    data_dict (dict): dict containing auditory nerve model output and all metadata
    hdf5_key_prefix (str): prefix added to hdf5 keys in data_key_pair_list
    data_keys (list): keys in data_dict that should be added to data_key_pair_list

    Returns
    -------
    data_key_pair_list (list): list of tuples (hdf5_key, data_key) for datasets with N rows
    '''
    data_key_pair_list = []
    for key in sorted(data_dict.keys()):
        if key in data_keys:
            data_key_pair_list.append((hdf5_key_prefix + key, key))
        else:
            for regex_key in [k for k in data_keys if '*' in k]:
                if re.match(regex_key, key):
                    data_key_pair_list.append((hdf5_key_prefix + key, key))
    return data_key_pair_list


def get_default_config_key_pair_list(data_dict,
                                     hdf5_key_prefix='config_bez2018model/',
                                     ignore_keys=['signal',
                                                  'pin',
                                                  'pin_dBSPL',
                                                  'nervegram_vihcs',
                                                  'nervegram_meanrates',
                                                  'nervegram_spike_times',
                                                  'nervegram_spike_tensor',
                                                  'nervegram_spike_tensor_dense',
                                                  'nervegram_spike_tensor_sparse*'],
                                     flat_keyparts=['_fs', '_list', '_shape']):
    '''
    Helper function to get default config_key_pair_list.

    Args
    ----
    data_dict (dict): dict containing auditory nerve model output and all metadata
    hdf5_key_prefix (str): prefix added to hdf5 keys in config_key_pair_list
    ignore_keys (list): keys in data_dict that should NOT be added to config_key_pair_list
    flat_keyparts (list): substrings indicating which config keys should be added without prefix

    Returns
    -------
    config_key_pair_list (list): list of tuples (hdf5_key, config_key) for config datasets
    '''
    config_key_pair_list = []
    for key in sorted(data_dict.keys()):
        is_config_key = True
        if key in ignore_keys:
            is_config_key = False
        for regex_key in [k for k in ignore_keys if '*' in k]:
            if re.match(regex_key, key):
                is_config_key = False
        if is_config_key:
            if any([keypart in key for keypart in flat_keyparts]):
                config_key_pair_list.append((key, key))
            else:
                config_key_pair_list.append((hdf5_key_prefix + key, key))
    return config_key_pair_list


def generate_nervegrams(hdf5_filename,
                        list_signal,
                        signal_fs,
                        list_snr=None,
                        list_noise=None,
                        list_dbspl=None,
                        disp_step=10,
                        data_key_pair_list=[],
                        config_key_pair_list=[],
                        kwargs_nervegram={},
                        kwargs_check_continuation={},
                        kwargs_initialization={}):
    '''
    Main routine for generating BEZ2018 ANmodel nervegrams and writing outputs to hdf5 file.

    Args
    ----
    hdf5_filename (str): filename for hdf5 dataset in which to store ANmodel outputs
    list_signal (np.ndarray): dataset of signal waveforms (signal index by time array)
    signal_fs (int): sampling rate of signals in list_signal (Hz)
    list_snr (np.ndarray): dataset of stimulus SNR values (if None, signal is used as stimulus)
    list_noise (np.ndarray): dataset of noise waveforms (required if list_snr is not None)
    list_dbspl (np.ndarray): dataset of stimulus dB SPL values (if None, signal level is used)
    disp_step (int): every disp_step, progress is displayed and hdf5 file is checkpointed
    data_key_pair_list (list): list of tuples (hdf5_key, data_key) for datasets with N rows
    config_key_pair_list (list): list of tuples (hdf5_key, config_key) for config datasets
    kwargs_nervegram (dict or list of dicts): kwargs for `bez2018model.nervegram()`
    kwargs_check_continuation (dict): kwargs for `check_continuation()`
    kwargs_initialization (dict): kwargs for `initialize_hdf5_file()`
    '''
    # Calculate total number of signals and convert kwargs_nervegram to list if needed
    N = list_signal.shape[0]
    if not isinstance(kwargs_nervegram, list):
        kwargs_nervegram = [kwargs_nervegram] * N
    assert len(kwargs_nervegram) == N, "nervegram parameter list must have length N"
    if list_snr is not None:
        assert len(list_snr) == N, "list_snr must have length N"
        assert list_noise is not None, "list_noise must be specified if list_snr is specified"
        assert list_noise.shape[0] == N, "list_noise must have length N"

    # Check if the hdf5 output dataset can be continued and get correct start_idx
    continuation_flag, start_idx = check_continuation(hdf5_filename, **kwargs_check_continuation)
    if start_idx is None:
        print('>>> [EXITING] No indexes remain in {}'.format(hdf5_filename))
        return

    # Open hdf5 file if continuing existing dataset
    if continuation_flag:
        print('>>> [CONTINUING] {} from index {}'.format(hdf5_filename, start_idx))
        hdf5_f = h5py.File(hdf5_filename, 'r+')

    # Main loop: iterate over all signals
    t_start = time.time()
    for idx in range(start_idx, N):
        # Preprocess input signal as specified
        signal = list_signal[idx]
        if list_snr is not None:
            snr = list_snr[idx]
            # De-mean signal and noise to ensure valid RMS computations
            signal = signal - np.mean(signal)
            noise = list_noise[idx] - np.mean(list_noise[idx])
            if np.isinf(snr) and snr > 0:
                # If SNR is positive infinity, do not add noise to signal
                signal = signal
            elif np.isinf(snr) and snr < 0:
                # If SNR is negative infinity, replace signal with noise
                signal = noise
            else:
                # Otherwise keep signal level fixed and add re-scaled noise
                rms_signal = np.sqrt(np.mean(np.square(signal)))
                rms_noise = np.sqrt(np.mean(np.square(noise)))
                rms_noise_scaling = rms_signal / (rms_noise * np.power(10, snr / 20))
                signal = signal + rms_noise_scaling * noise
        # Update stimulus sound presentation level settings in kwargs_nervegram
        if list_dbspl is not None:
            kwargs_update = {
                'pin_dBSPL_flag': 1,
                'pin_dBSPL': list_dbspl[idx],
            }
            kwargs_nervegram[idx] = recursive_dict_merge(kwargs_nervegram[idx], kwargs_update)

        # Run stimulus through ANmodel and generate meanrates nervegram
        data_dict = bez2018model.nervegram(signal, signal_fs, **kwargs_nervegram[idx])

        # If key pair lists are empty, get reasonable defaults
        if len(data_key_pair_list) == 0:
            print('>>> [WARNING] Using default data_key_pair_list')
            data_key_pair_list = get_default_data_key_pair_list(data_dict)
        if len(config_key_pair_list) == 0:
            print('>>> [WARNING] Using default config_key_pair_list')
            config_key_pair_list = get_default_config_key_pair_list(data_dict)

        # If output hdf5 file dataset has not been initialized, do so on first iteration
        if not continuation_flag:
            print('>>> [INITIALIZING] {}'.format(hdf5_filename))
            assert idx == 0, "hdf5 dataset should only be initialized when idx=0"
            initialize_hdf5_file(hdf5_filename,
                                 N,
                                 data_dict,
                                 data_key_pair_list=data_key_pair_list,
                                 config_key_pair_list=config_key_pair_list,
                                 **kwargs_initialization)
            continuation_flag = True
            hdf5_f = h5py.File(hdf5_filename, 'r+')
            for (k, _) in data_key_pair_list:
                print('>>> [DATA] {}'.format(k), hdf5_f[k])
            for (k, _) in config_key_pair_list:
                print('>>> [CONFIG] {}'.format(k), hdf5_f[k])

        # Write the ANmodel outputs to the hdf5 dataset
        write_example_to_hdf5(hdf5_f,
                              data_dict,
                              idx,
                              data_key_pair_list=data_key_pair_list)

        # Display progress and close/re-open hdf5 dataset
        if idx % disp_step == 0:
            hdf5_f.close()
            hdf5_f = h5py.File(hdf5_filename, 'r+')
            t_mean_per_signal = (time.time() - t_start) / (idx - start_idx + 1) # Seconds per signal
            t_est_remaining = (N - idx - 1) * t_mean_per_signal / 60.0 # Estimated minutes remaining
            disp_str = ('... signal {:06d} of {:06d} | dB SPL: {:02.2f} | '
                        'time_mean_per_signal: {:02.2f} sec | time_est_remaining: {:06.0f} min')
            print(disp_str.format(idx, N, data_dict['pin_dBSPL'], t_mean_per_signal, t_est_remaining))

    # Close the hdf5 dataset for the last time
    hdf5_f.close()
    print('>>> [COMPLETING] {}'.format(hdf5_filename))


def run_dataset_generation(source_hdf5_filename,
                           dest_hdf5_filename,
                           idx_start=0,
                           idx_end=None,
                           source_key_signal='/signal',
                           source_key_signal_fs='/signal_rate',
                           source_key_snr=None,
                           source_key_noise=None,
                           source_key_dbspl=None,
                           source_keys_to_copy=[],
                           range_dbspl=None,
                           **kwargs):
    '''
    Read stimuli from hdf5 file, generate ANmodel nervegrams, and copy specified datasets
    from source to destination hdf5 file.

    Args
    ----
    source_hdf5_filename (str): filename for hdf5 dataset providing the ANmodel inputs
    dest_hdf5_filename (str): filename for hdf5 dataset in which to store ANmodel outputs
    idx_start (int): specifies first stimulus in source_hdf5_filename to process
    idx_end (int or None): upper limit of stimulus range in source_hdf5_filename to process
    source_key_signal (str): key for signal dataset in source_hdf5_filename
    source_key_signal_fs (str): key for signal sampling rate in source_hdf5_filename
    source_key_snr (str): key for stimulus SNR dataset (source_key_noise is ignored if None)
    source_key_noise (str): key for noise dataset in source_hdf5_filename (required if source_key_snr is specified)
    source_key_dbspl (str): key for stimulus dB SPL dataset in source_hdf5_filename
    source_keys_to_copy (list): keys for datasets in source_hdf5_filename to copy to dest_hdf5_filename
    range_dbspl (list): min and max sound presentation level (only used if source_key_dbspl is None)
    **kwargs (passed directly to `generate_nervegrams()`)
    '''
    # Ensure source and destination filenames are different and open the source hdf5 file
    assert not source_hdf5_filename == dest_hdf5_filename, "source and dest hdf5 files must be different"
    source_hdf5_f = h5py.File(source_hdf5_filename, 'r')

    # Collect input signals and sampling rate from the source hdf5 file
    if (idx_end == None) or (idx_end > source_hdf5_f[source_key_signal].shape[0]):
        idx_end = source_hdf5_f[source_key_signal].shape[0]
    assert idx_start < idx_end, "idx_start must be less than idx_end"
    list_signal = source_hdf5_f[source_key_signal][idx_start:idx_end]
    signal_fs = source_hdf5_f[source_key_signal_fs][0]
    # Collect input noise and SNR from the source hdf5 file
    if source_key_snr is not None:
        list_snr = source_hdf5_f[source_key_snr][idx_start:idx_end]
        msg = "source_key_noise must be specified if source_key_snr is specified"
        assert source_key_noise is not None, msg
        list_noise = source_hdf5_f[source_key_noise][idx_start:idx_end]
    else:
        list_snr = None
        list_noise = None
    # Collect stimulus sound presentation levels from the source hdf5 file
    if source_key_dbspl is not None:
        assert range_dbspl is None, "cannot specifiy both source_key_dbspl and range_dbspl"
        list_dbspl = source_hdf5_f[source_key_dbspl][idx_start:idx_end]
    elif range_dbspl is not None:
        assert len(range_dbspl) == 2, "range_dbspl must be (min, max) pair"
        list_dbspl = np.random.uniform(low=range_dbspl[0], high=range_dbspl[1], size=list_signal.shape[0])
    else:
        list_dbspl = None

    # Run the main ANmodel nervegram generation routine
    print('>>> [START] {}'.format(dest_hdf5_filename))
    generate_nervegrams(dest_hdf5_filename,
                        list_signal,
                        signal_fs,
                        list_snr=list_snr,
                        list_noise=list_noise,
                        list_dbspl=list_dbspl,
                        **kwargs)

    # Copy specified datasets from source hdf5 file to destination hdf5 file
    dsets_to_copy = {}
    for key in source_keys_to_copy:
        if key in source_hdf5_f:
            if source_hdf5_f[key].shape[0] == 1:
                key_dset = source_hdf5_f[key]
            else:
                key_dset = source_hdf5_f[key][idx_start:idx_end]
            dsets_to_copy[key] = dask.array.from_array(key_dset, chunks=key_dset.shape)
            print('>>> [COPYING FROM SOURCE]: {}'.format(key), dsets_to_copy[key].shape)
    if dsets_to_copy:
        dask.array.to_hdf5(dest_hdf5_filename, dsets_to_copy)

    # Close the source hdf5 file
    source_hdf5_f.close()
    print('>>> [END] {}'.format(dest_hdf5_filename))


def parallel_run_dataset_generation(source_regex,
                                    dest_filename,
                                    job_idx=0,
                                    jobs_per_source_file=10,
                                    source_key_signal='/signal',
                                    **kwargs):
    '''
    Wrapper function to easily parallelize `run_dataset_generation()`.

    Args
    ----
    source_regex (str): regular expression that globs all source hdf5 filenames
    dest_filename (str): filename for output hdf5 file (indexes will be added for parallel outputs)
    job_idx (int): index of current job
    jobs_per_source_file (int): number of jobs each source file is split into
    source_key_signal (str): key for signal dataset in source_hdf5_filename
    **kwargs (passed directly to `generate_nervegrams()`)
    '''
    # Determine the source_hdf5_filename using source_regex, job_idx, and jobs_per_source_file
    source_fn_list = sorted(glob.glob(source_regex))
    assert len(source_fn_list) > 0, "source_regex did not match any files"
    source_file_idx = job_idx // jobs_per_source_file
    assert source_file_idx < len(source_fn_list), "source_file_idx out of range"
    source_hdf5_filename = source_fn_list[source_file_idx]

    # Compute idx_start and idx_end within source_hdf5_filename for the given job_idx
    source_hdf5_f = h5py.File(source_hdf5_filename, 'r')
    N = source_hdf5_f[source_key_signal].shape[0]
    idx_splits = np.linspace(0, N, jobs_per_source_file + 1, dtype=int)
    idx_start = idx_splits[job_idx % jobs_per_source_file]
    idx_end = idx_splits[(job_idx % jobs_per_source_file) + 1]
    source_hdf5_f.close()

    # Design unique destination hdf5 filename
    sidx = dest_filename.rfind('.')
    if len(source_fn_list) == 1:
        dest_hdf5_filename = dest_filename[:sidx] + '_{:06d}-{:06d}' + dest_filename[sidx:]
        dest_hdf5_filename = dest_hdf5_filename.format(idx_start, idx_end)
    else:
        dest_hdf5_filename = dest_filename[:sidx] + '_{:03}_{:06d}-{:06d}' + dest_filename[sidx:]
        dest_hdf5_filename = dest_hdf5_filename.format(source_file_idx, idx_start, idx_end)

    # Call `run_dataset_generation()` to launch the nervegram generation routine
    print('>>> [PARALLEL_RUN] job_idx: {}, source_file_idx: {} of {}, jobs_per_source_file: {}'.format(
        job_idx, source_file_idx, len(source_fn_list), jobs_per_source_file))
    print('>>> [PARALLEL_RUN] source_hdf5_filename: {}'.format(source_hdf5_filename))
    print('>>> [PARALLEL_RUN] dest_hdf5_filename: {}'.format(dest_hdf5_filename))
    run_dataset_generation(source_hdf5_filename,
                           dest_hdf5_filename,
                           idx_start=idx_start,
                           idx_end=idx_end,
                           source_key_signal=source_key_signal,
                           **kwargs)
