clear all;

numcfs = 30;
CFs   = logspace(log10(300),log10(20e3),numcfs);  % CF in Hz;

numsponts_healthy = [0 0 4];
numsponts = numsponts_healthy(3);

if exist('ANpopulation.mat','file')
    load('ANpopulation.mat');
    disp('Loading existing population of AN fibers saved in ANpopulation.mat')
    if (size(sponts.LS,2)<numsponts_healthy(1))||(size(sponts.MS,2)<numsponts_healthy(2))||(size(sponts.HS,2)<numsponts_healthy(3))||(size(sponts.HS,1)<numcfs||~exist('tabss','var'))
        disp('Saved population of AN fibers in ANpopulation.mat is too small - generating a new population');
        [sponts,tabss,trels] = generateANpopulation(numcfs,numsponts_healthy);
    end
else
    [sponts,tabss,trels] = generateANpopulation(numcfs,numsponts_healthy);
    disp('Generating population of AN fibers, saved in ANpopulation.mat')
end

numstims = 55;

rates = zeros(numcfs,numsponts,numstims);

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
nrep = 10;  % number of stimulus repetitions - Liberman (1978) used 10;

t = 0:1/Fs:T-1/Fs; % time vector
mxpts = length(t);
irpts = rt*Fs;


for cflp = 1:numcfs
    
    CF = CFs(cflp);
    
    F0 = CF;  % stimulus frequency in Hz
    
    for spontlp = 1:numsponts
        
        spont = sponts.HS(cflp,spontlp);
        tabs  = tabss.HS(cflp,spontlp);
        trel  = trels.HS(cflp,spontlp);
        
        thrsh = find_CF_Threshold_BEZ2018(CF,Fs,cohc,cihc,species,noiseType,implnt,spont,tabs,trel);
        
        stimdbs = thrsh:thrsh+54;
        
        for stimlp = 1:numstims
            
            disp(['CFlp = ' int2str(cflp) '/' int2str(numcfs) '; spontlp = ' int2str(spontlp) '/' int2str(sum(numsponts)) '; stimlp = ' int2str(stimlp) '/' int2str(sum(numstims))])
            
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
        
    end
    
end

figure
semilogx(CFs/1e3,max(rates,[],3),'ko')
xlim([0.1 40])
ylim([100 350])
xlabel('Characteristic Frequency (kHz)')
ylabel('Discharge Rate at Saturation (/s)')


