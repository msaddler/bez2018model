import sys
import os
import bez2018model
import pdb
import numpy as np
import h5py
import dask.array as da
import glob


''' NOT YET FUNCTIONAL '''


def write_example_to_hdf5(hdf5_f, out, idx, key_pair_list=[]):
    '''
    Write individual example to open hdf5 file.
    
    Args
    ----
    hdf5_f (h5py.File object): open and writeable hdf5 file object
    out (dict): dict containing auditory nerve model output
    idx (int): specifies row of hdf5 file to write to
    key_pair_list (list): list of tuples (hdf5_key, out_key)
    '''
    for (hdf5_key, out_key) in key_pair_list: hdf5_f[hdf5_key][idx] = np.array(out[out_key])


def initialize_hdf5_file(hdf5_filename, N, out, out_key_pair_list=[], config_key_pair_list=[]):
    '''
    Create a new hdf5 file and populate config params (overwrite if file exists).
    
    Args
    ----
    hdf5_filename (str): filename for new hdf5 file
    N (int): number of examples that will be written to file
    out (dict): dict containing auditory nerve model output
    out_key_pair_list (list): list of tuples (hdf5_key, out_key) for datasets with `N` rows
    config_key_pair_list (list): list of tuples (hdf5_key, out_key) for config datasets
    '''
    # Initialize hdf5 file
    f = h5py.File(hdf5_filename, 'w')
    # Create the main output datasets
    for (hdf5_key, out_key) in out_key_pair_list:
        out_key_value = np.squeeze(np.array(out[out_key])).astype(np.float32)
        out_key_shape = [N] + list(out_key_value.shape)
        f.create_dataset(hdf5_key, out_key_shape, dtype=out_key_value.dtype)
    # Create and populate teh config datasets
    for (hdf5_key, out_key) in config_key_pair_list:
        out_key_value = np.squeeze(np.array(out[out_key])).astype(np.float32)
        out_key_shape = [1] + list(out_key_value.shape)
        f.create_dataset(hdf5_key, out_key_shape, dtype=out_key_value.dtype, data=out_key_value)
    # Close the initialized hdf5 file
    f.close()


def generate_nervegrams(dest_filename, signal_list, signal_Fs, disp_step=10,
                        out_key_pair_list=[], config_key_pair_list=[],
                        output_params={}, ANmodel_params={}, manipulation_params={}):
    '''
    Runs the auditory nerve model and stores auditory nervegrams in `dest_filename`.
    
    Args
    ----
    
    Returns
    -------
    '''
    
    ### Determine if hdf5 file needs to be initialized and get start_index
    try:
        f = h5py.File(dest_filename, 'r+') # Attempt to open the output hdf5 file
        start_index = np.reshape(np.argwhere(f['/pin_dBSPL'][:] == 0), (-1,))
        if len(start_index) == 0: # Quit if file is complete
            f.close()
            print('>>> FILE FOUND: no indexes remain')
            return
        else:
            start_index = np.max([0, start_index[0]-1]) # Start 1 signal before to be safe
            print('>>> FILE FOUND: starting at index {}'.format(start_index))
        init_flag = False
    except: # Initialize new hdf5 file if the specified one does not exist
        init_flag = True
        start_index = 0
    
    ### Start MATLAB engine and run each signal through the auditory nerve model
    eng = bez2018model.start_matlab_engine()
    N = signal_list.shape[0]
    for idx in range(start_index, N):
        # Compute bez2018model output on every iteration
        out = bez2018model.generate_nervegram(eng, signal_list[idx], signal_Fs, output_params,
                                              ANmodel_params, manipulation_params)
        # Initialize the hdf5 file on the first iteration
        if init_flag:
            print('>>> INITIALIZING:', dest_filename)
            initialize_hdf5_file(dest_filename, N, out,
                                 out_key_pair_list=out_key_pair_list,
                                 config_key_pair_list=config_key_pair_list)
            f = h5py.File(dest_filename, 'r+')
            init_flag = False
        # Write the bez2018model output to hdf5 file on every iteration
        write_example_to_hdf5(f, out, idx, key_pair_list=out_key_pair_list)
        # Close and re-open hdf5 file to checkpoint every `disp_step` iterations
        if idx % disp_step == 0:
            f.close()
            f = h5py.File(dest_filename, 'r+')
            print('... signal {} of {}'.format(idx, N))
    
    ### Close the hdf5 file and quit the MATLAB engine
    f.close()
    bez2018model.quit_matlab_engine(eng)


def get_ERB_CF_list(num_CFs, min_CF=125., max_CF=10e3):
    '''
    Helper function to get array of num_CFs ERB-scaled CFs between min_CF and max_CF.
    '''
    E_start = 21.4 * np.log10(0.00437 * min_CF + 1.0)
    E_end = 21.4 * np.log10(0.00437 * max_CF + 1.0)
    CF_list = np.linspace(E_start, E_end, num = num_CFs)
    CF_list = (1.0/0.00437) * (10.0 ** (CF_list / 21.4) - 1.0)
    return list(CF_list)


def get_bez2018model_ANmodel_params(num_CFs=40, min_CF=125., max_CF=8e3, spont_list=[70.],
                                    cohc=1., cihc=1., species=2.):
    '''
    Helper function to get reasonable ANmodel_params for bez2018model.
    '''
    ANmodel_params = {}
    ANmodel_params['CF_list'] = get_ERB_CF_list(num_CFs, min_CF=min_CF, max_CF=max_CF)
    ANmodel_params['spont_list'] = spont_list
    ANmodel_params['cohc'] = cohc
    ANmodel_params['cihc'] = cihc
    ANmodel_params['species'] = species
    return ANmodel_params


def get_bez2018model_output_params(meanrates_dur=2., meanrates_Fs=10e3, set_dBSPL_flag=0,
                                   buffer_front_dur=0.070, buffer_end_dur=0.010):
    '''
    Helper function to get reasonable output_params for bez2018model.
    '''
    output_params = {}
    output_params['meanrates_dur'] = meanrates_dur
    output_params['meanrates_Fs'] = meanrates_Fs
    output_params['set_dBSPL_flag'] = set_dBSPL_flag
    output_params['buffer_front_dur'] = buffer_front_dur
    output_params['buffer_end_dur'] = buffer_end_dur
    return output_params


def get_bez2018model_manipulation_params():
    '''
    Helper function to get reasonable manipulation_params for bez2018model.
    '''
    manipulation_params = {}
    return manipulation_params


def get_config_key_pair_list(ANmodel_params, output_params):
    '''
    Helper function to get config_key_pair_list.
    '''
    config_key_pair_list = []
    for key in list(ANmodel_params.keys()) + list(output_params.keys()):
        config_key_pair_list.append(('/config_ANmodel/'+key, key))
    return config_key_pair_list


def get_out_key_pair_list():
    '''
    Helper function to get out_key_pair_list.
    '''
    out_key_pair_list = []
    for key in ['meanrates', 'meanrates_clip_indexes', 'pin_dBSPL']:
        out_key_pair_list.append(('/'+key, key))
    return out_key_pair_list


def main(source_f, dest_filename, idx_start=0, idx_end=None, disp_step=5,
         source_f_signal_list_key='/pin', source_f_signal_Fs_key='/signal_rate',
         kwargs_ANmodel_params={}, kwargs_output_params={}, kwargs_manipulation_params={}):
    
    # Use the helper functions to get the arguments for `generate_nervegrams`
    ANmodel_params = get_bez2018model_ANmodel_params(**kwargs_ANmodel_params)
    output_params = get_bez2018model_output_params(**kwargs_output_params)
    manipulation_params = get_bez2018model_manipulation_params(**kwargs_manipulation_params)
    out_key_pair_list = get_out_key_pair_list()
    config_key_pair_list = get_config_key_pair_list(ANmodel_params, output_params)
    
    # Collect the signal inputs and sampling rate for `generate_nervegrams`
    if (idx_end == None) or (idx_end > source_f[source_f_signal_list_key].shape[0]):
        idx_end = source_f[source_f_signal_list_key].shape[0]
    assert idx_start < idx_end, 'idx_start must be less than idx_end'
    signal_list = source_f[source_f_signal_list_key][idx_start : idx_end]
    signal_Fs = source_f[source_f_signal_Fs_key][0]
    
    # Collect the datasets that will be copied directly from `source_f`
    keys_to_copy = ['/noise_condition', '/phone_labels_unaligned', '/phoneme_labels_unaligned', '/snr',
                    source_f_signal_Fs_key, source_f_signal_list_key]
    dsets_to_copy = {}
    for key in keys_to_copy:
        if key in source_f:
            if source_f[key].shape[0] == 1: key_dset = source_f[key]
            else: key_dset = source_f[key][idx_start : idx_end]
            dsets_to_copy[key] = da.from_array(key_dset, chunks=key_dset.shape)
    
    # Call function to run the auditory nerve model and write outputs to `dest_filename`
    generate_nervegrams(dest_filename, signal_list, signal_Fs, disp_step=disp_step,
                        out_key_pair_list=out_key_pair_list,
                        config_key_pair_list=config_key_pair_list,
                        output_params=output_params,
                        ANmodel_params=ANmodel_params,
                        manipulation_params=manipulation_params)
    
    # Copy the datasets specified earlier directly from `source_f` to `dest_filename`
    for key in dsets_to_copy: print('COPYING: {}'.format(key), dsets_to_copy[key].shape)
    da.to_hdf5(dest_filename, dsets_to_copy)
    source_f.close()


if __name__ == "__main__":
    
    assert len(sys.argv) >= 3, 'Required command line arguments are: <job_id> <N>'
    job_id = int(sys.argv[1])
    N = int(sys.argv[2])
    
    # Determine source filename and start_idx / end_idx from source_regex and dset_start_idx
#     source_regex = ('/om2/user/msaddler/hearing_impaired_networks/timit/sr20000/train_snr-10to+10/'
#                     'timitWithAudiosetNoise_*.hdf5')
#     source_regex = '/om2/user/msaddler/hearing_impaired_networks/timit/sr20000/test_behavioral_validation/timitWithAudiosetNoise_snr-10to+10_*.hdf5'
    source_regex = '/om2/user/msaddler/hearing_impaired_networks/timit/sr20000/test_behavioral/*.hdf5'
    
    ### CALCULATE INDEXES FOR TRAINING / VALIDATION DATA
#     source_fn_list = sorted(glob.glob(source_regex))
#     source_file_bin_starts = []
#     source_file_bin_ends = []
#     dset_start_idx = job_id * N
#     for source_fn in source_fn_list:
#         [dsi, dei] = [int(x) for x in os.path.basename(source_fn)[-18:-5].split('-')]
#         source_file_bin_starts.append(dsi)
#         source_file_bin_ends.append(dei)
#     source_file_idx = np.digitize(dset_start_idx, source_file_bin_starts) - 1
#     source_fn = source_fn_list[source_file_idx]
#     idx_start = dset_start_idx - source_file_bin_starts[source_file_idx]
#     max_idx_end = source_file_bin_ends[source_file_idx] - source_file_bin_starts[source_file_idx]
#     idx_end = min(idx_start + N, max_idx_end)
    
    ### CALCULATE INDEXES FOR TEST BEHAVIORAL
    job_idx = job_id
    source_fn_list = sorted(glob.glob(source_regex))
    examples_per_source_file = 3460
    jobs_per_source_file = np.ceil(examples_per_source_file/N).astype(int)
    source_file_idx = job_idx // jobs_per_source_file
    idx_start = N * (job_idx % jobs_per_source_file)
    idx_end = min(idx_start + N, examples_per_source_file)
    dset_start_idx = idx_start
    source_fn = source_fn_list[source_file_idx]
    
    # Load the source file and name the destination file
    print('<SOURCE>', source_fn)
    print('<SOURCE idx_start, idx_end>', idx_start, idx_end)
    source_f = h5py.File(source_fn, 'r')
    
    ### DESTINATION FILENAME FOR TEST BEHAVIORAL
    dest_dir = '/om/scratch/Wed/msaddler/data_6345proj/timit/sr20000/bez2018_coch1_test_behavioral/'
    if len(sys.argv) == 4:
        dest_dir = str(sys.argv[3])
        print('using command line specified dest_dir:', dest_dir)
    dest_filename = os.path.join(dest_dir, os.path.basename(source_fn))
    dest_filename = dest_filename.replace(dest_filename[-18:-5], '{:06}-{:06}')
    
    ### DESTINATION FILENAME FOR TRAINING / VALIDATION
#     dest_filename = ('/om/scratch/Wed/msaddler/data_6345proj/timit/sr20000/bez2018_coch0_train_snr-10to+10/'
#                      'timitWithAudiosetNoise_{:06}-{:06}.hdf5')
    
    dest_filename = dest_filename.format(dset_start_idx, dset_start_idx+idx_end-idx_start)
    if 'coch0' in dest_filename:
        kwargs_ANmodel_params={'cohc':0.0} # <--- QUICK PATCH to set ANmodel_params
        print('kwargs_ANmodel_params', kwargs_ANmodel_params)
    else:
        kwargs_ANmodel_params={'cohc':1.}
        print('kwargs_ANmodel_params', kwargs_ANmodel_params)
    print('<START>', dest_filename)
    main(source_f, dest_filename, idx_start=idx_start, idx_end=idx_end, disp_step=5,
         source_f_signal_list_key='/pin', source_f_signal_Fs_key='/signal_rate',
         kwargs_ANmodel_params=kwargs_ANmodel_params, kwargs_output_params={}, kwargs_manipulation_params={})
    print('<END>', dest_filename)
