clear all;

numcfs = 1;
CFs   = 8e3;  % CF in Hz;

% numsponts = [10 10 30]; % reduce the number of ANFs to have a shorter simulation time
numsponts = [20 20 60];

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

stimdbs = -10:5:100;
numstims = length(stimdbs);
rates = zeros(numcfs,sum(numsponts),numstims);

cohc  = 1.0;   % normal ohc function
cihc  = 1.0;   % normal ihc function
species = 1;   % 1 for cat (2 for human with Shera et al. tuning; 3 for human with Glasberg & Moore tuning)
implnt = 0;    % "0" for approximate or "1" for actual implementation of the power-law functions in the Synapse
noiseType = 1; % 1 for variable fGn; 0 for fixed (frozen) fGn
Fs = 100e3;    % sampling rate in Hz (must be 100, 200 or 500 kHz)

% stimulus parameters
T  = 50e-3;  % stimulus duration in seconds
rt = 2.5e-3; % rise/fall time in seconds

% PSTH parameters
psthbinwidth = 0.5e-3; % binwidth in seconds;
nrep = 100;  % number of stimulus repetitions - Liberman (1978) used 10;

t = 0:1/Fs:T-1/Fs; % time vector
mxpts = length(t);
irpts = rt*Fs;

figure

for cflp = 1:numcfs
    
    CF = CFs(cflp);
    
    F0 = CF;  % stimulus frequency in Hz
    
    for spontlp = 1:sum(numsponts)
        
        sponts_concat = [sponts.LS(cflp,1:numsponts(1)) sponts.MS(cflp,1:numsponts(2)) sponts.HS(cflp,1:numsponts(3))];
        tabss_concat = [tabss.LS(cflp,1:numsponts(1)) tabss.MS(cflp,1:numsponts(2)) tabss.HS(cflp,1:numsponts(3))];
        trels_concat = [trels.LS(cflp,1:numsponts(1)) trels.MS(cflp,1:numsponts(2)) trels.HS(cflp,1:numsponts(3))];
        
        spont = sponts_concat(spontlp);
        tabs = tabss_concat(spontlp);
        trel = trels_concat(spontlp);
        
        for stimlp = 1:numstims
            
            disp(['cflp = ' int2str(cflp) '/' int2str(numcfs) '; spontlp = ' int2str(spontlp) '/' int2str(sum(numsponts)) '; stimlp = ' int2str(stimlp) '/' int2str(sum(numstims))])
            
            % flush the output for the display of the coutput in Octave
            if exist ('OCTAVE_VERSION', 'builtin') ~= 0
                fflush(stdout);
            end
            
            stimdb = stimdbs(stimlp);
            pin = sqrt(2)*20e-6*10^(stimdb/20)*sin(2*pi*F0*t); % unramped stimulus
            pin(1:irpts) = pin(1:irpts).*(0:(irpts-1))/irpts;
            pin((mxpts-irpts):mxpts) = pin((mxpts-irpts):mxpts).*(irpts:-1:0)/irpts;
            
            vihc = model_IHC_BEZ2018(pin,CF,nrep,1/Fs,T*2,cohc,cihc,species);
            psth = model_Synapse_BEZ2018(vihc,CF,nrep,1/Fs,noiseType,implnt,spont,tabs,trel);
            
            timeout = (0:length(psth)-1)*1/Fs;
            psthbins = round(psthbinwidth*Fs);  % number of psth bins per psth bin
            psthtime = timeout(1:psthbins:end); % time vector for psth
            pr = sum(reshape(psth,psthbins,length(psth)/psthbins))/nrep; % pr of spike in each bin
            psTH = pr/psthbinwidth; % psth in units of /s
            
            ronset = round(1.5e-3/psthbinwidth)+1;
            roffset = round(T/psthbinwidth);
            
            rates(cflp,spontlp,stimlp)= mean(psTH(ronset:ronset+roffset));
            
        end
        
        if (spontlp<=numsponts(1))
            plot(stimdbs,squeeze(rates(cflp,spontlp,:)),'r')
            hold on
        elseif (spontlp<=sum(numsponts([1 2])))
            plot(stimdbs,squeeze(rates(cflp,spontlp,:)),'b')
            hold on
        else
            plot(stimdbs,squeeze(rates(cflp,spontlp,:)),'m')
            hold on
        end
        
    end
    
end

xlabel('Stimulus Level (dB SPL)')
ylabel('Firing Rate (/s)')




