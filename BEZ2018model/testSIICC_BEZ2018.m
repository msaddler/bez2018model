clear all;

% model fiber parameters
cohc  = 1.0;   % normal ohc function
cihc  = 1.0;   % normal ihc function
species = 1;   % 1 for cat (2 for human with Shera et al. tuning; 3 for human with Glasberg & Moore tuning)
noiseType = 1; % 1 for variable fGn; 0 for fixed (frozen) fGn
implnt = 0;    % "0" for approximate or "1" for actual implementation of the power-law functions in the Synapse

trials = 500;

numcfs = 30;

CFs   = logspace(log10(300),log10(20e3),numcfs);  % CF in Hz;

numsponts = [0 10 30];

if exist('ANpopulation.mat','file')
    load('ANpopulation.mat');
    disp('Loading existing population of AN fibers saved in ANpopulation.mat')
    if (size(sponts.LS,2)<numsponts(1))||(size(sponts.MS,2)<numsponts(2))||(size(sponts.HS,2)<numsponts(3))||(size(sponts.HS,1)<numcfs||~exist('tabss','var'))
        disp('Saved population of AN fibers in ANpopulation.mat is too small - generating a new population');
        [sponts,tabss,trels] = generateANpopulation(numcfs,numsponts);
    end
else
    [sponts,tabss,trels] = generateANpopulation(numcfs,numsponts);
    disp('Generating population of AN fibers, saved in ANpopulation.mat')
end


% stimulus parameters
Fs = 100e3;  % sampling rate in Hz (must be 100, 200 or 500 kHz)
T  = 20;  % stimulus duration in seconds

% PSTH parameters
nrep = 1;               % number of stimulus repetitions (e.g., 50);
psthbinwidth = 0.5e-3; % binwidth in seconds;

t = 0:1/Fs:T-1/Fs; % time vector

vihc = zeros(1,length(t));

trial=1;
m = zeros(1,trials);
p = zeros(1,trials);

meanrate = [];

while trial<=trials
    
    fprintf(1,'trial = %i/%i',trial,trials);
    
    % flush the output for the display of the coutput in Octave
    if exist ('OCTAVE_VERSION', 'builtin') ~= 0
      fflush(stdout);
    end
    
    cfind = ceil(numcfs*rand(1));
    
    CF = CFs(cfind);
    
    sponts_concat = [sponts.LS(cfind,1:numsponts(1)) sponts.MS(cfind,1:numsponts(2)) sponts.HS(cfind,1:numsponts(3))];
    tabss_concat = [tabss.LS(cfind,1:numsponts(1)) tabss.MS(cfind,1:numsponts(2)) tabss.HS(cfind,1:numsponts(3))];
    trels_concat = [trels.LS(cfind,1:numsponts(1)) trels.MS(cfind,1:numsponts(2)) trels.HS(cfind,1:numsponts(3))];
    
    spontind = ceil(sum(numsponts)*rand(1));
    
    spont = sponts_concat(spontind);
    tabs = tabss_concat(spontind);
    trel = trels_concat(spontind);
                                               
    psth = model_Synapse_BEZ2018(vihc,CF,nrep,1/Fs,noiseType,implnt,spont,tabs,trel);
    
    sptimes= find (psth==1)/Fs;
    nspikes=length(sptimes);
    
    ISI = diff(sptimes); % Compute ISIs from spike times
    N = length(ISI);
        
    if N>500
        [m(trial),p(trial)] = compute_SIICC(ISI, N);
        trial= trial+1;
        display([':  ' num2str(N) ' spikes in this trial']);
        if exist ('OCTAVE_VERSION', 'builtin') ~= 0
            fflush(stdout);
        end

    else
        display([':  only ' num2str(N) ' spikes in this trial - rerunning trial']);
        if exist ('OCTAVE_VERSION', 'builtin') ~= 0
            fflush(stdout);
        end
    end
end


figure
plot(m*1e3,p,'.',m*1e3,0*ones(1,length(m)))
xlim([0 50]); ylim([-0.2 0.2])
xlabel('Mean ISI (ms)')
ylabel('SIICC')
grid





