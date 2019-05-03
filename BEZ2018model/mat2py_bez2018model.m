function out = mat2py_bez2018model(signal, signal_Fs,...
    output_params, ANmodel_params, manipulation_params)
% MS 2018.12.07
% - MATLAB function to generate auditory nervegram using BEZ2018 ANmodel
% - Designed to be called from MATLAB engine Python API
% - 'meanrates' output = CF-by-TIME-by-FIBERTYPE (spont rate) tensor
%
% INPUTS:
%   signal: single channel input waveform
%   signal_Fs: sampling rate (Hz) of signal
%   output_params (struct / Python dict)
%   ANmodel_params (struct / Python dict)
%   manipulation_params (struct / Python dict)
% RETURNS:
%   out (struct / Python dict)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PARSE INPUTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% output_params struct
meanrates_dur = 0.050; % Nervegram duration in (s)
if isfield(output_params, 'meanrates_dur')
    meanrates_dur = output_params.meanrates_dur;
end
meanrates_Fs = 10e3; % Nervegram sampling rate in (Hz)
if isfield(output_params, 'meanrates_Fs')
    meanrates_Fs = output_params.meanrates_Fs;
end
buffer_front_dur = 0.070; % Default = delete first 70ms of nervegram
if isfield(output_params, 'buffer_front_dur')
    buffer_front_dur = output_params.buffer_front_dur;
end
buffer_end_dur = 0.010; % Default = delete last 10ms of nervegram
if isfield(output_params, 'buffer_end_dur')
    buffer_end_dur = output_params.buffer_end_dur;
end
set_dBSPL_flag = 1; % Flag sets whether or not pin is scaled to pin_dBSPL 
if isfield(output_params, 'set_dBSPL_flag')
    set_dBSPL_flag = output_params.set_dBSPL_flag;
end
pin_dBSPL = 65; % Default dB SPL (only enforced if flag is set) 
if isfield(output_params, 'pin_dBSPL')
    pin_dBSPL = output_params.pin_dBSPL;
end

% ANmodel_params struct
CF_list = ANmodel_params.CF_list; % CFs (Hz) for each fiber
spont_list = ANmodel_params.spont_list; % Spont rate list (fiber types)
cohc = ANmodel_params.cohc;
cihc = ANmodel_params.cihc;
species = ANmodel_params.species;

% manipulation_params struct (add cochlear manipulations here)
manipulation_flag = 0; % Flag sets whether or not manipulations are applied
if isfield(manipulation_params, 'manipulation_flag')
    manipulation_flag = manipulation_params.manipulation_flag;
end
filt_cutoff = 0;
if isfield(manipulation_params, 'filt_cutoff')
    filt_cutoff = manipulation_params.filt_cutoff;
end
filt_order = 6;
if isfield(manipulation_params, 'filt_order')
    filt_order = manipulation_params.filt_order;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PREPARE INPUT SIGNAL FOR ANMODEL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if size(signal, 1) ~= 1
    signal = signal';
end
pin_Fs = 100e3; % sampling rate for ANmodel (100kHz)
pin_dur = length(signal) / signal_Fs; % signal duration (s)
pin = resample(signal, pin_Fs, signal_Fs); % resample for ANmodel
if set_dBSPL_flag
    P_rms = 2e-5 * 10^(pin_dBSPL / 20); % pressure (Pa)
    pin = P_rms * (pin / rms(pin)); % set dB SPL
else
    pin_dBSPL = 20 * log10(rms(pin) / 2e-5);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% RUN ANMODEL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Model fiber parameters
tabs = 0.6e-3; % Absolute refractory period (s)
trel = 0.6e-3; % Baseline mean relative refractory period (s)
noiseType = 1; % 1 for variable fGn (0 for fixed fGn)
implnt = 0; % 0=approximate; 1=actual implementation of the power-law fcns

% Model time parameters
T = length(pin) / pin_Fs; % duration of signal (s)
nrep = 1; % number of stimulus repetitions
dt = 1 / pin_Fs; % step size (s)
reptime = T + 1/pin_Fs; % time between stim repetitions (s), must be >= T

% Initialize output variables (CF-by-TIME-by-FIBER tensor)
meanrates_len = floor(pin_dur * meanrates_Fs);
meanrates = zeros(length(CF_list), meanrates_len, length(spont_list));

for itrC = 1:length(CF_list) % Iterate over CFs
    CF = CF_list(itrC);
    % Run inner hair cell (IHC) model %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    vihc = MS_model_IHC_BEZ2018(...
        pin, CF, nrep, dt, reptime, cohc, cihc, species);
    
    for itrF = 1:length(spont_list) % Iterate over spont rates
        spont = spont_list(itrF);
        % Run inner hair cell (IHC) model %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [~, meanrate, ~, ~, ~,~] = MS_model_Synapse_BEZ2018(...
            vihc, CF, nrep, dt, noiseType, implnt, spont, tabs, trel);
    
        % Store downsampled output of AN synapse model
        meanrate = resample(meanrate, meanrates_Fs, pin_Fs);
        meanrate(meanrate < 0) = 0; % Remove negative resampling artifacts
        meanrates(itrC, :, itrF) = meanrate(1 : meanrates_len);
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% MODIFY AND CLIP ANMODEL OUTPUT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if manipulation_flag
    if filt_cutoff > 0 % Low-pass filter the auditory nervegram
        Wn = filt_cutoff / (meanrates_Fs / 2); % Cutoff frequency
        [b, a] = butter(filt_order, Wn, 'low'); % Get filter coefficients
        for itrF = 1:length(spont_list)
            tmp = filtfilt(b, a, meanrates(:, :, itrF)')';
            meanrates(:, :, itrF) = tmp;
        end
    end
end

% Clip the ends of full-length nervegram to interval of specified length
meanrates_clipped_len = meanrates_dur * meanrates_Fs;
clip_start = 1;
clip_end = clip_start + meanrates_clipped_len - 1;
if meanrates_clipped_len < meanrates_len
    buffer_front = ceil(buffer_front_dur * meanrates_Fs);
    buffer_end = meanrates_len - floor(buffer_end_dur * meanrates_Fs);
    clip_start = randi([buffer_front, buffer_end - meanrates_clipped_len]);
    clip_end = clip_start + meanrates_clipped_len - 1;
    assert(clip_end <= buffer_end, 'meanrates clip_end is out of buffered range')
    meanrates = meanrates(:, clip_start:clip_end, :);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% STORE OUTPUTS IN STRUCTURE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Store basic signal parameters
out.signal_Fs = signal_Fs;
% Store ANmodel input signal and parameters
out.pin = pin;
out.pin_Fs = pin_Fs;
out.pin_dur = pin_dur;
out.pin_dBSPL = pin_dBSPL;
out.pin_clip_indexes = round([clip_start,clip_end] * pin_Fs/meanrates_Fs);
% ANmodel outputs and parameters
out.meanrates = meanrates;
out.meanrates_Fs = meanrates_Fs;
out.meanrates_dur = meanrates_dur;
out.meanrates_clip_indexes = [clip_start, clip_end];
out.meanrates_filt_order = filt_order;
out.meanrates_filt_cutoff = filt_cutoff;
% ANmodel specified parameters
out.ANmodel_CF_list = CF_list;
out.ANmodel_spont_list = spont_list;
out.ANmodel_cohc = cohc;
out.ANmodel_cihc = cihc;
out.ANmodel_species = species;
out.ANmodel_noiseType = noiseType;
out.ANmodel_implnt = implnt;

end
