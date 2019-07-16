% model_IHC_BEZ2018 - Bruce, Erfani & Zilany (2018) Auditory Nerve Model
%
%     vihc = model_IHC_BEZ2018(pin,CF,nrep,dt,reptime,cohc,cihc,species);
%
% vihc is the inner hair cell (IHC) relative transmembrane potential (in volts)
%
% pin is the input sound wave in Pa sampled at the appropriate sampling rate (see instructions below)
% CF is the characteristic frequency of the fiber in Hz
% nrep is the number of repetitions for the psth
% dt is the binsize in seconds, i.e., the reciprocal of the sampling rate (see instructions below)
% reptime is the time between stimulus repetitions in seconds - NOTE should be equal to or longer than the duration of pin
% cohc is the OHC scaling factor: 1 is normal OHC function; 0 is complete OHC dysfunction
% cihc is the IHC scaling factor: 1 is normal IHC function; 0 is complete IHC dysfunction
% species is the model species: "1" for cat, "2" for human with BM tuning from Shera et al. (PNAS 2002),
%    or "3" for human BM tuning from Glasberg & Moore (Hear. Res. 1990)
%
% For example,
%
%    vihc = model_IHC_BEZ2018(pin,1e3,10,1/100e3,0.2,1,1,2); **requires 8 input arguments
%
% models a normal human fiber of high spontaneous rate (normal OHC & IHC function) with a CF of 1 kHz, 
% for 10 repetitions and a sampling rate of 100 kHz, for a repetition duration of 200 ms, and
% with approximate implementation of the power-law functions in the synapse model.
%
%
% NOTE ON SAMPLING RATE:-
% Since version 2 of the code, it is possible to run the model at a range
% of sampling rates between 100 kHz and 500 kHz.
% It is recommended to run the model at 100 kHz for CFs up to 20 kHz, and
% at 200 kHz for CFs> 20 kHz to 40 kHz.