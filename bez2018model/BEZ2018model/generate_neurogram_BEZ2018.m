function [neurogram_ft,neurogram_mr,neurogram_Sout,t_ft,t_mr,t_Sout,CFs] = generate_neurogram_BEZ2018(stim,Fs_stim,species,ag_fs,ag_dbloss)

% model fiber parameters
numcfs = 40;
CFs   = logspace(log10(250),log10(16e3),numcfs);  % CF in Hz;

% cohcs  = ones(1,numcfs);  % normal ohc function
% cihcs  = ones(1,numcfs);  % normal ihc function

dbloss = interp1(ag_fs,ag_dbloss,CFs,'linear','extrap');

% mixed loss
[cohcs,cihcs,OHC_Loss]=fitaudiogram2(CFs,dbloss,species);

% OHC loss
% [cohcs,cihcs,OHC_Loss]=fitaudiogram(CFs,dbloss,species,dbloss);

% IHC loss
% [cohcs,cihcs,OHC_Loss]=fitaudiogram(CFs,dbloss,species,zeros(size(CFs)));

numsponts_healthy = [10 10 30]; % Number of low-spont, medium-spont, and high-spont fibers at each CF in a healthy AN

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

implnt = 0;    % "0" for approximate or "1" for actual implementation of the power-law functions in the Synapse
noiseType = 1;  % 0 for fixed fGn (1 for variable fGn)

% stimulus parameters
Fs = 100e3;  % sampling rate in Hz (must be 100, 200 or 500 kHz)
stim100k = resample(stim,Fs,Fs_stim).';
T  = length(stim100k)/Fs;  % stimulus duration in seconds

% PSTH parameters
nrep = 1;
psthbinwidth_mr = 100e-6; % mean-rate binwidth in seconds;
windur_ft=32;
smw_ft = hamming(windur_ft);
windur_mr=128;
smw_mr = hamming(windur_mr);

pin = stim100k(:).';

clear stim100k

simdur = ceil(T*1.2/psthbinwidth_mr)*psthbinwidth_mr;

for CFlp = 1:numcfs
    
    
    CF = CFs(CFlp);
    cohc = cohcs(CFlp);
    cihc = cihcs(CFlp);
    
    numsponts = round([1 1 1].*numsponts_healthy); % Healthy AN
    %     numsponts = round([0.5 0.5 0.5].*numsponts_healthy); % 50% fiber loss of all types
    %     numsponts = round([0 1 1].*numsponts_healthy); % Loss of all LS fibers
    %     numsponts = round([cihc 1 cihc].*numsponts_healthy); % loss of LS and HS fibers proportional to IHC impairment
    
    sponts_concat = [sponts.LS(CFlp,1:numsponts(1)) sponts.MS(CFlp,1:numsponts(2)) sponts.HS(CFlp,1:numsponts(3))];
    tabss_concat = [tabss.LS(CFlp,1:numsponts(1)) tabss.MS(CFlp,1:numsponts(2)) tabss.HS(CFlp,1:numsponts(3))];
    trels_concat = [trels.LS(CFlp,1:numsponts(1)) trels.MS(CFlp,1:numsponts(2)) trels.HS(CFlp,1:numsponts(3))];
    
    vihc = model_IHC_BEZ2018(pin,CF,nrep,1/Fs,simdur,cohc,cihc,species);
    
    for spontlp = 1:sum(numsponts)
        
        disp(['CFlp = ' int2str(CFlp) '/' int2str(numcfs) '; spontlp = ' int2str(spontlp) '/' int2str(sum(numsponts))])

        % flush the output for the display of the coutput in Octave
        if exist ('OCTAVE_VERSION', 'builtin') ~= 0
            fflush(stdout);
        end
        
        
        spont = sponts_concat(spontlp);
        tabs = tabss_concat(spontlp);
        trel = trels_concat(spontlp);
        
        [psth_ft,meanrate,varrate,synout] = model_Synapse_BEZ2018(vihc,CF,nrep,1/Fs,noiseType,implnt,spont,tabs,trel);
        psthbins = round(psthbinwidth_mr*Fs);  % number of psth_ft bins per psth bin
        psth_mr = sum(reshape(psth_ft,psthbins,length(psth_ft)/psthbins));
        
        if spontlp == 1
            neurogram_ft(CFlp,:) = filter(smw_ft,1,psth_ft);
            neurogram_Sout(CFlp,:) = synout;            
            neurogram_mr(CFlp,:) = filter(smw_mr,1,psth_mr);
        else
            
            neurogram_ft(CFlp,:) = neurogram_ft(CFlp,:)+filter(smw_ft,1,psth_ft);
            neurogram_Sout(CFlp,:) = neurogram_Sout(CFlp,:)+synout;
            neurogram_mr(CFlp,:) = neurogram_mr(CFlp,:)+filter(smw_mr,1,psth_mr);
        end
        
    end
    
end

neurogram_ft = neurogram_ft(:,1:windur_ft/2:end); % 50% overlap in Hamming window
t_ft = 0:windur_ft/2/Fs:(size(neurogram_ft,2)-1)*windur_ft/2/Fs; % time vector for the fine-timing neurogram
neurogram_mr = neurogram_mr(:,1:windur_mr/2:end); % 50% overlap in Hamming window
t_mr = 0:windur_mr/2*psthbinwidth_mr:(size(neurogram_mr,2)-1)*windur_mr/2*psthbinwidth_mr; % time vector for the mean-rate neurogram
t_Sout = 0:1/Fs:(size(neurogram_Sout,2)-1)/Fs; % time vector for the synapse output neurogram

