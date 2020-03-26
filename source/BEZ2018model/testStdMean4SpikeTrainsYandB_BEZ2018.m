clear all;

% Check to see if running under Matlab or Octave
if exist ('OCTAVE_VERSION', 'builtin') ~= 0
    %   pkg load signal;
    if exist('rms')<1
        rms = @(x) sqrt(mean(x.^2));
    end
end

numcfs = 10;
CFs   = logspace(log10(300),log10(20e3),numcfs);  % CF in Hz;

numsponts = [2 2 6];

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

stimdbs = 0:5:60;
numstims = length(stimdbs);

cohc  = 1.0;   % normal ohc function
cihc  = 1.0;   % normal ihc function
species = 1;   % 1 for cat (2 for human with Shera et al. tuning; 3 for human with Glasberg & Moore tuning)
implnt = 0;    % "0" for approximate or "1" for actual implementation of the power-law functions in the Synapse
noiseType = 1; % 1 for variable fGn; 0 for fixed (frozen) fGn
Fs = 100e3;    % sampling rate in Hz (must be 100, 200 or 500 kHz)

% stimulus parameters
T  = 2;  % stimulus duration in seconds
rt = 10e-3; % rise/fall time in seconds

% PSTH parameters
nrep = 1;               % number of stimulus repetitions (e.g., 50);
psthbinwidth = 1.25e-3; % binwidth in seconds;
psthbins = round(psthbinwidth*Fs);  % number of psth bins per psth bin

t = 0:1/Fs:T-1/Fs; % time vector
mxpts = length(t);
irpts = rt*Fs;

trials = 50;

cnt_noise = zeros(numcfs,sum(numsponts),numstims,trials);

for cflp = 1:numcfs
    
    CF = CFs(cflp);
    
    sponts_concat = [sponts.LS(cflp,1:numsponts(1)) sponts.MS(cflp,1:numsponts(2)) sponts.HS(cflp,1:numsponts(3))];
    tabss_concat = [tabss.LS(cflp,1:numsponts(1)) tabss.MS(cflp,1:numsponts(2)) tabss.HS(cflp,1:numsponts(3))];
    trels_concat = [trels.LS(cflp,1:numsponts(1)) trels.MS(cflp,1:numsponts(2)) trels.HS(cflp,1:numsponts(3))];
    
    
    for spontlp = 1:sum(numsponts)
        
        spont = sponts_concat(spontlp);
        tabs = tabss_concat(spontlp);
        trel = trels_concat(spontlp);
        
        
        for stimlp = 1:numstims
            
            stimdb = stimdbs(stimlp);
            
            pin = randn(size(t)); % Generate a white Gaussian noise stimulus
            pin = 20e-6*10^(stimdb/20)*pin(1:length(t))/rms(pin(1:length(t))); % unramped stimulus
            pin(1:irpts)= pin(1:irpts).*(0:(irpts-1))/irpts;
            pin((mxpts-irpts):mxpts)=pin((mxpts-irpts):mxpts).*(irpts:-1:0)/irpts;
            
            vihc = model_IHC_BEZ2018(pin,CF,nrep,1/Fs,T,cohc,cihc,species);
            
            for trial = 1:trials
                
                disp(['cflp = ' int2str(cflp) '/' int2str(numcfs) '; spontlp = ' int2str(spontlp) '/' int2str(sum(numsponts)) '; stimlp = ' int2str(stimlp) '/' int2str(sum(numstims)) '; trial = ' int2str(trial) '/' int2str(trials)])
                
                % flush the output for the display of the coutput in Octave
                if exist ('OCTAVE_VERSION', 'builtin') ~= 0
                    fflush(stdout);
                end
                
                psth = model_Synapse_BEZ2018(vihc,CF,nrep,1/Fs,noiseType,implnt,spont,tabs,trel);
                Psth = sum(reshape(psth,psthbins,length(psth)/psthbins));
                
                tvect = 0:psthbinwidth:(length(Psth)-1)*psthbinwidth;
                tstart = find(tvect>=1.650,1,'first');
                tend = find(tvect>=1.850,1,'first');
                
                cnt_noise(cflp,spontlp,stimlp,trial) = sum(Psth(tstart:tend)); % Calculate the spike count in the period 1.25 - 51.25 ms
                
            end
        end
        
    end
    
end

mean_cnt = mean(cnt_noise,4);
std_cnt  = std(cnt_noise,0,4);

mean_cnt_LS = mean_cnt(:,1:numsponts(1),:);
std_cnt_LS = std_cnt(:,1:numsponts(1),:);
mean_cnt_MS = mean_cnt(:,1+numsponts(1):numsponts(1)+numsponts(2),:);
std_cnt_MS = std_cnt(:,1+numsponts(1):numsponts(1)+numsponts(2),:);
mean_cnt_HS = mean_cnt(:,1+numsponts(1)+numsponts(2):sum(numsponts),:);
std_cnt_HS = std_cnt(:,1+numsponts(1)+numsponts(2):sum(numsponts),:);

m = linspace(0,40,100);
s = sqrt(m);

figure
plot(mean_cnt_LS(mean_cnt_LS<35),std_cnt_LS(mean_cnt_LS<35),'k^','markersize',4)
hold on
plot(mean_cnt_MS(mean_cnt_MS<35),std_cnt_MS(mean_cnt_MS<35),'ks','markersize',4)
plot(mean_cnt_HS(mean_cnt_HS<35),std_cnt_HS(mean_cnt_HS<35),'kx','markersize',6)
plot(m,s,'k-')
xlim([0 40]);
ylim([0 6])
xlabel('Mean spike count for 200ms')
ylabel('Standard deviation in spike count for 200ms')
title('Cf. Fig. 6a of Young and Barta (1986)')
grid
