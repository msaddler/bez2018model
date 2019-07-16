clear all; clc

% model parameters
CF    = 5e3;   % CF in Hz;
spont = 100;   % spontaneous firing rate
tabs   = 0.6e-3; % Absolute refractory period
trel   = 0.6e-3; % Baseline mean relative refractory period
cohc  = 1.0;    % normal ohc function
cihc  = 1.0;    % normal ihc function
species = 1;    % 1 for cat (2 for human with Shera et al. tuning; 3 for human with Glasberg & Moore tuning)
noiseType = 1;  % 1 for variable fGn; 0 for fixed (frozen) fGn
implnt = 0;     % "0" for approximate or "1" for actual implementation of the power-law functions in the Synapse

% stimulus parameters
stimdb = 60; % stimulus intensity in dB SPL
F0 = CF;     % stimulus frequency in Hz
Fs = 100e3;  % sampling rate in Hz (must be 100, 200 or 500 kHz)
T  = 0.25;  % stimulus duration in seconds
rt = 2.5e-3; % rise/fall time in seconds
ondelay = 25e-3;
trials = 1e3;

% PSTH parameters
nrep = 1;               % number of stimulus repetitions (e.g., 50);
psthbinwidth = 5e-4; % binwidth in seconds;
psthbins = round(psthbinwidth*Fs);  % number of psth bins per psth bin

t = 0:1/Fs:T-1/Fs; % time vector
mxpts = length(t);
irpts = rt*Fs;
onbin = round(ondelay*Fs);

pin = zeros(1,onbin+mxpts);

pin(onbin+1:onbin+mxpts) = sqrt(2)*20e-6*10^(stimdb/20)*sin(2*pi*F0*t); % unramped stimulus
pin(onbin+1:onbin+irpts)= pin(onbin+1:onbin+irpts).*(0:(irpts-1))/irpts;
pin(onbin+(mxpts-irpts):onbin+mxpts)=pin(onbin+(mxpts-irpts):onbin+mxpts).*(irpts:-1:0)/irpts;

dt=1/Fs; %  time step

vihc = model_IHC_BEZ2018(pin,CF,nrep,dt,2*T,cohc,cihc,species);

trial = 1;

disp(['trial = ' num2str(trial) '/' num2str(trials)])

% flush the output for the display of the coutput in Octave
if exist ('OCTAVE_VERSION', 'builtin') ~= 0
    fflush(stdout);
end

[psth,meanrate,varrate,synout, trd_vector,trel_vector] = model_Synapse_BEZ2018(vihc,CF,nrep,dt,noiseType,implnt,spont,tabs,trel);

synout_vectors = zeros(trials,length(trd_vector));
trel_vectors = zeros(trials,length(trd_vector));
trd_vectors = zeros(trials,length(trd_vector));
trd_means = zeros(trials,1);
trd_stds = zeros(trials,1);

trd_means(1,:) = mean(trd_vector);
trd_stds(1,:) = std(trd_vector);
trd_vectors(1,:) = trd_vector;
trel_vectors(1,:) = trel_vector;
synout_vectors(1,:) = synout;

Psth = sum(reshape(psth,psthbins,length(psth)/psthbins)); %

cnt = zeros(trials,length(Psth));

cnt(trial,:) = cumsum(Psth,2); % Calculate the cumulative spike count in each trial
sp (trial, : ) = Psth;

for trial = 2:trials
    
    disp(['trial = ' num2str(trial) '/' num2str(trials)])
    
    % flush the output for the display of the coutput in Octave
    if exist ('OCTAVE_VERSION', 'builtin') ~= 0
        fflush(stdout);
    end
    
    
    [psth,meanrate,varrate,synout,trd_vector,trel_vector] = model_Synapse_BEZ2018(vihc,CF,nrep,dt,noiseType,implnt,spont,tabs,trel);
    
    Psth = sum(reshape(psth,psthbins,length(psth)/psthbins)); %
    
    cnt(trial,:) = cumsum(Psth,2); % Calculate the cumulative spike count in each trial
    sp (trial, : ) = Psth;
    trd_means(trial,:) = mean(trd_vector);
    trd_stds(trial,:) = std(trd_vector);
    trd_vectors(trial,:) = trd_vector;
    trel_vectors(trial,:) = trel_vector;
    synout_vectors(trial,:) = synout;
    
end

simtime = length(psth)/Fs;

tvect = 0:psthbinwidth:simtime-psthbinwidth;

figure
bar(tvect, sum(sp)/trials/psthbinwidth,'histc') % Plot of estimated mean spike rate
ylabel('Firing Rate (/s)')
xlabel('Time (s)')
title('PSTH')

tt= 0:1/Fs:(length(trd_vector)-1)/Fs;

figure
errorbar(tt(1:500:end),mean(trd_vectors(:,1:500:end))*1e3,std(trd_vectors(:,1:500:end))*1e3);
ylabel('t_{rd} (ms)')
xlabel('Time (s)')

figure
subplot(3,1,1)
plot(tt,synout_vectors(1:50,:));
ylabel('S_{out} (/s)')
subplot(3,1,2)
plot(tt(1:500:end),trd_vectors(1:50,1:500:end)*1e3);
ylabel('\tau_{rd} (ms)')
subplot(3,1,3)
plot(tt,trel_vectors(1:50,:)*1e3);
ylabel('t_{rel} (ms)')
xlabel('Time (s)')
