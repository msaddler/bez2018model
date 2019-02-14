function [stimdb] = find_CF_Threshold_BEZ2018(CF,Fs,cohc,cihc,species,noiseType,implnt,spont,tabs,trel)
% find CF Threshold using the STB tone and incremental intensity by 1dB until when the firing rate for
% a specific fiber passed 10 plus the spontrate of the fiber

stimdb = -10;
F0=CF;
psthbinwidth = 0.5e-3; % binwidth in seconds;
nrep = 200;  % number of stimulus repetitions - Liberman (1978) used 10;
T  = 50e-3;  % stimulus duration in seconds
rt = 2.5e-3; % rise/fall time in seconds

t = 0:1/Fs:T-1/Fs; % time vector
mxpts = length(t);
irpts = rt*Fs;

SpontRate = spont;

firingRate_Icreased_to  = SpontRate;

while ((firingRate_Icreased_to  <(SpontRate + 10)) && (stimdb < 50))
    stimdb =  stimdb+1;
    if exist ('OCTAVE_VERSION', 'builtin') ~= 0
        fflush(stdout);
    end
    pin = sqrt(2)*20e-6*10^(stimdb/20)*sin(2*pi*F0*t); % unramped stimulus
    pin(1:irpts) = pin(1:irpts).*(0:(irpts-1))/irpts;
    pin((mxpts-irpts):mxpts) = pin((mxpts-irpts):mxpts).*(irpts:-1:0)/irpts;
    
    vihc = model_IHC_BEZ2018(pin,CF,nrep,1/Fs,T*2,cohc,cihc,species);
    psth = model_Synapse_BEZ2018(vihc,CF,nrep,1/Fs,noiseType,implnt,spont,tabs,trel);
    
    psthbins = round(psthbinwidth*Fs);  % number of psth bins per psth bin
    pr = sum(reshape(psth,psthbins,length(psth)/psthbins))/nrep; % pr of spike in each bin
    psTH = pr/psthbinwidth; % psth in units of spikes/s
    
    ronset =  round(1.5e-3/psthbinwidth)+1;
    roffset = round(T/psthbinwidth);
    
    SpontRate = mean(psTH(roffset+1:end));
    
    firingRate_Icreased_to = mean(psTH(ronset:ronset+roffset));
    
end