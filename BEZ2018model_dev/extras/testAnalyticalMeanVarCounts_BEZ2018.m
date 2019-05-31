clear all;

% model parameters
CF    = 8e3;   % CF in Hz;
cohc  = 1.0;    % normal ohc function
cihc  = 1.0;    % normal ihc function
species = 1;    % 1 for cat (2 for human with Shera et al. tuning; 3 for human with Glasberg & Moore tuning)
noiseType = 1;   % 1 for variable fGn; 0 for fixed (frozen) fGn
spont = 100; % spontaneous firing rate 
tabs   = 0.6e-3; % Absolute refractory period
trel   = 0.6e-3; % Baseline mean relative refractory period
implnt = 0;     % "0" for approximate or "1" for actual implementation of the power-law functions in the Synapse

% stimulus parameters
F0 = CF;     % stimulus frequency in Hz
Fs = 100e3;  % sampling rate in Hz (must be 100, 200 or 500 kHz)
T  = 0.25;  % stimulus duration in seconds
rt = 2.5e-3; % rise/fall time in seconds
stimdb = 20; % stimulus intensity in dB SPL

% PSTH parameters
nrep = 1;               % number of stimulus repetitions (e.g., 50);
psthbinwidths = [5e-4 5e-3 5e-2]; % binwidth in seconds;
numpsthbinwidths = length(psthbinwidths);
trials = 1e3;
% trials = 10e3; % higher number of trials gives more accurate estimates but takes longer to run

t = 0:1/Fs:T-1/Fs; % time vector
mxpts = length(t);
irpts = rt*Fs;

ondelay = 25e-3;

onbin = round(ondelay*Fs);

pin = zeros(1,onbin+mxpts);

pin(onbin+1:onbin+mxpts) = sqrt(2)*20e-6*10^(stimdb/20)*sin(2*pi*F0*t); % unramped stimulus
pin(onbin+1:onbin+irpts)= pin(onbin+1:onbin+irpts).*(0:(irpts-1))/irpts;
pin(onbin+(mxpts-irpts):onbin+mxpts)=pin(onbin+(mxpts-irpts):onbin+mxpts).*(irpts:-1:0)/irpts;

vihc = model_IHC_BEZ2018(pin,CF,nrep,1/Fs,2*T,cohc,cihc,species);

disp(['lp = ' int2str(1) '/' int2str(trials)])

% Check to see if running under Matlab or Octave
if exist ('OCTAVE_VERSION', 'builtin') ~= 0
    fflush(stdout);
end

[psth,meanrate,varrate,synout,trd_vector,trel_vector] = model_Synapse_BEZ2018(vihc,CF,nrep,1/Fs,noiseType,implnt,spont,tabs,trel);

timeout = (0:length(psth)-1)*1/Fs;

psthbins = zeros(numpsthbinwidths,1);
psthtime = cell(numpsthbinwidths);
psths = cell(numpsthbinwidths);
mrs = cell(numpsthbinwidths);
vrs = cell(numpsthbinwidths);

for binlp = 1:numpsthbinwidths
    
    psthbins(binlp) = round(psthbinwidths(binlp)*Fs);  % number of psth bins per psth bin
    psthtime{binlp} = timeout(1:psthbins(binlp):end); % time vector for psth
    cnt = sum(reshape(psth,psthbins(binlp),length(psth)/psthbins(binlp))); % spike cnt in each psth bin
    mr = mean(reshape(meanrate,psthbins(binlp),length(psth)/psthbins(binlp))); % mean average of theor spike cnt in each psth bin
    vr = mean(reshape(varrate,psthbins(binlp),length(psth)/psthbins(binlp))); % mean var of theor spike cnt in each psth bin
    
    psths{binlp}(1,:) = cnt;
    mrs{binlp}(1,:) = mr;
    vrs{binlp}(1,:) = vr;
    
end

for lp = 2:trials
    
    disp(['lp = ' int2str(lp) '/' int2str(trials)])
    if exist ('OCTAVE_VERSION', 'builtin') ~= 0
        fflush(stdout);
    end
    
    [psth,meanrate,varrate,synout,trd_vector,trel_vector] = model_Synapse_BEZ2018(vihc,CF,nrep,1/Fs,noiseType,implnt,spont,tabs,trel);
    
    timeout = (0:length(psth)-1)*1/Fs;
    
    for binlp = 1:numpsthbinwidths
        
        psthbins(binlp) = round(psthbinwidths(binlp)*Fs);  % number of psth bins per psth bin
        psthtime{binlp} = timeout(1:psthbins(binlp):end); % time vector for psth
        cnt = sum(reshape(psth,psthbins(binlp),length(psth)/psthbins(binlp))); % spike cnt in each psth bin
        mr = mean(reshape(meanrate,psthbins(binlp),length(psth)/psthbins(binlp))); % mean average of theor spike cnt in each psth bin
        vr = mean(reshape(varrate,psthbins(binlp),length(psth)/psthbins(binlp))); % mean var of theor spike cnt in each psth bin
        
        psths{binlp}(lp,:) = cnt;
        mrs{binlp}(lp,:) = mr;
        vrs{binlp}(lp,:) = vr;
        
    end
    
end

for binlp = 1:numpsthbinwidths
    
    if psthbinwidths(binlp)>10e-3
        mrksize = 6;
    else
        mrksize = 2;
    end
    figure
    subplot(2,1,1)
    h1 = bar(psthtime{binlp},mean(psths{binlp}),'histc');
    set(h1,'edgecolor','k','facecolor',0.8*ones(1,3))
    ylabel('E[count]')
    xlabel('Time (s)')
    hold on
    plot(psthtime{binlp}+psthbinwidths(binlp)/2,mean(mrs{binlp})*psthbinwidths(binlp),'ro','markerfacecolor','r','markersize',mrksize,'linewidth',1)
    xlim(psthtime{binlp}([1 end]))
    title(['PSTH bin width = ' num2str(psthbinwidths(binlp)*1e3,2) 'ms'])
    subplot(2,1,2)
    h2 = bar(psthtime{binlp},var(psths{binlp}),'histc');
    set(h2,'edgecolor','k','facecolor',0.8*ones(1,3))
    xlabel('Time (s)')
    ylabel('var[count]')
    hold on
    plot(psthtime{binlp}+psthbinwidths(binlp)/2,mean(vrs{binlp})*psthbinwidths(binlp),'go','markerfacecolor','g','markersize',mrksize,'linewidth',1)
    plot(psthtime{binlp}+psthbinwidths(binlp)/2,mean(mrs{binlp})*psthbinwidths(binlp),'bo','markerfacecolor','b','markersize',mrksize,'linewidth',1)
    xlim(psthtime{binlp}([1 end]))
    
end


