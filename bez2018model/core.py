import sys
import matlab.engine
import numpy as np


def start_matlab_engine():
    print('Starting matlab engine ...')
    return matlab.engine.start_matlab()


def quit_matlab_engine(eng):
    print('Quitting matlab engine ...')
    eng.quit()


def convert_arrays_to_numpy(out):
    '''
    Scans fields of provided dictionary and converts all
    MATLAB arrays to numpy arrays.
    
    Args
    ----
    out (dict): array fields contain MATLAB arrays
    
    Returns
    -------
    out (dict): array fields contain numpy arrays
    '''
    for key in out.keys():
        val = out.get(key)
        if hasattr(val, 'size') and hasattr(val, '_data'):
            out[key] = np.array(val._data).reshape(val.size, order='F')
    return out


def convert_arrays_to_matlab(param_dict):
    '''
    Scans fields of provided dictionary and converts all
    numpy arrays to MATLAB arrays.
    
    Args
    ----
    out (dict): array fields contain numpy arrays
    
    Returns
    -------
    out (dict): array fields contain MATLAB arrays
    '''
    for key in param_dict.keys():
        param_dict[key] = matlab.double([param_dict[key]])
    return param_dict


def generate_nervegram(eng, signal, signal_Fs, output_params, ANmodel_params, manipulation_params):
    '''
    Function takes in matlab engine, signal, and parameter set
    and returns a dictionary containing a BEZ2018 nervegram.
    
    Args
    ----
    eng (matlab engine)
    signal (array): input signal, must be single channel
    signal_Fs (int): input signal sampling rate
    output_params (dict): see required fields in `mat2py_bez2018model.m`
    ANmodel_params (dict): see required fields in `mat2py_bez2018model.m`
    manipulation_params (dict): see required fields in `mat2py_bez2018model.m`
    
    Returns
    -------
    out (dict): fields are define in `mat2py_bez2018model.m`
    '''
    # Cast inputs into MATLAB-acceptable datatypes
    signal = matlab.double(list(signal))
    signal_Fs = float(signal_Fs)
    output_params = convert_arrays_to_matlab(output_params)
    ANmodel_params = convert_arrays_to_matlab(ANmodel_params)
    manipulation_params = convert_arrays_to_matlab(manipulation_params)
    # Run the BEZ2018 Auditory nerve model
    out = eng.mat2py_bez2018model(signal, signal_Fs, output_params,
                                  ANmodel_params, manipulation_params, nargout=1)
    # Cast output fields to numpy arrays and return
    out = convert_arrays_to_numpy(out)
    return out