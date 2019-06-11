clear all;

% model fiber parameters
CF    = 1.5e3;   % CF in Hz;
spont = 50;      % spontaneous firing rate
tabs   = 0.6e-3; % absolute refractory period
trel   = 0.6e-3; % baseline mean relative refractory period
cohc  = 1.0;     % normal ohc function
cihc  = 1.0;     % normal ihc function
species = 1;     % 1 for cat (2 for human with Shera et al. tuning; 3 for human with Glasberg & Moore tuning)
noiseType = 1;   % 1 for variable fGn; 0 for fixed (frozen) fGn
implnt = 0;      % "0" for approximate or "1" for actual implementation of the power-law functions in the Synapse

% stimulus parameters
F0 = CF;     % stimulus frequency in Hz
Fs = 100e3;  % sampling rate in Hz (must be 100, 200 or 500 kHz)
T  = 25;  % stimulus duration in seconds
rt = 2.5e-3; % rise/fall time in seconds

stimdb = -inf; % stimulus intensity in dB SPL; set to -inf to get spont activity

trials = 10;

numTs = 14;
dt=1/Fs; %  each time in length
Ts = logspace(log10(1e-3),log10(10),numTs);
Ts = round(Ts/dt)*dt;

Ft = zeros(trials,numTs);
Ft_shuf = zeros(trials,numTs);
meanrate = zeros(trials,numTs);

% PSTH parameters

nrep = 1;               % number of stimulus repetitions (e.g., 50);
t = 0:1/Fs:T-1/Fs; % time vector
mxpts = length(t);
irpts = rt*Fs;

pin = sqrt(2)*20e-6*10^(stimdb/20)*sin(2*pi*F0*t); % unramped stimulus
pin(1:irpts)= pin(1:irpts).*(0:(irpts-1))/irpts;
pin((mxpts-irpts):mxpts)=pin((mxpts-irpts):mxpts).*(irpts:-1:0)/irpts;

vihc = model_IHC_BEZ2018(pin,CF,nrep,1/Fs,2*T,cohc,cihc,species);

for trial = 1:trials
    
    disp(['trial = ' num2str(trial) '/' num2str(trials)])
    % flush the output for the display of the coutput in Octave
    if exist ('OCTAVE_VERSION', 'builtin') ~= 0
        fflush(stdout);
    end
    
    
    psth = model_Synapse_BEZ2018(vihc,CF,nrep,1/Fs,noiseType,implnt,spont,tabs,trel);
    
    simtime = length(psth)/Fs;
    tvect = 0:1/Fs:simtime-1/Fs;
    
    ISIs = diff(tvect(logical(psth)));
    ISIs_shuf = ISIs(randperm(length(ISIs)));
    spiketimes_shuf = cumsum(ISIs_shuf);
    psth_shuf = histc(spiketimes_shuf,tvect);
    
    for Tlp = 1:numTs
        
        psthbinwidth = Ts(Tlp);
        psthbins = round(psthbinwidth*Fs);  % number of time bins per Psth bin
        numPsthbins = floor(length(psth)/psthbins);
        
        Psth = sum(reshape(psth(1:psthbins*numPsthbins),psthbins,numPsthbins)); %
        
        Psth_shuf = sum(reshape(psth_shuf(1:psthbins*numPsthbins),psthbins,numPsthbins));
        
        Ft(trial,Tlp) = std(Psth)^2/(mean(Psth)+eps);
        Ft_shuf(trial,Tlp) = std(Psth_shuf)^2/(mean(Psth_shuf)+eps);
        meanrate(trial,Tlp) = mean(Psth)/Ts(Tlp);
        
    end
    
end

figure
loglog(Ts*1e3,Ft) % Plot the calculated Fano factor curve for each trial
hold on
loglog(Ts*1e3,mean(Ft),'k-','linewidth',2) % Plot the mean Fano factor curve
loglog(Ts*1e3,Ft_shuf,'--') % Plot the calculated Fano factor curves for the shuffled ISIs for each trial
loglog(Ts*1e3,mean(Ft_shuf),'k--','linewidth',2) % Plot the calculated Fano factor curves for the shuffled ISIs for each trial
ylabel('F(T)')
xlabel('T (ms)')
xlim([1e0 1e4])
ylim([0.2 10])
