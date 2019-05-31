clear all;

numcfs = 20;
% numcfs = 40; % increase to get a better sampling of CFs

CFs   = logspace(log10(125),log10(15e3),numcfs);  % CF in Hz (range to check BTC vs CF)
numsponts = [0 0 15]; % Just use high spont fibers to check BTC vs CF

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

cohc  = 1.0;   % normal ohc function
cihc  = 1.0;   % normal ihc function
species = 1;   % 1 for cat (2 for human with Shera et al. tuning; 3 for human with Glasberg & Moore tuning)
implnt = 0;    % "0" for approximate or "1" for actual implementation of the power-law functions in the Synapse
noiseType = 1; % 1 for variable fGn; 0 for fixed (frozen) fGn
Fs = 100e3;    % sampling rate in Hz (must be 100, 200 or 500 kHz)

thrsh = zeros(numcfs,sum(numsponts));
thrsh_meanHS = zeros(numcfs,1);
thrsh_stdHS = zeros(numcfs,1);

for cflp = 1:numcfs
    
    CF = CFs(cflp);
    
    F0 = CF;  % stimulus frequency in Hz
    
    for spontlp = 1:sum(numsponts)
        
        disp(['CFlp = ' int2str(cflp) '/' int2str(numcfs) '; spontlp = ' int2str(spontlp) '/' int2str(sum(numsponts))])
        
        % flush the output for the display of the coutput in Octave
        if exist ('OCTAVE_VERSION', 'builtin') ~= 0
            fflush(stdout);
        end
        
        sponts_concat = [sponts.LS(cflp,1:numsponts(1)) sponts.MS(cflp,1:numsponts(2)) sponts.HS(cflp,1:numsponts(3))];
        tabss_concat = [tabss.LS(cflp,1:numsponts(1)) tabss.MS(cflp,1:numsponts(2)) tabss.HS(cflp,1:numsponts(3))];
        trels_concat = [trels.LS(cflp,1:numsponts(1)) trels.MS(cflp,1:numsponts(2)) trels.HS(cflp,1:numsponts(3))];
        
        spont = sponts_concat(spontlp);
        tabs = tabss_concat(spontlp);
        trel = trels_concat(spontlp);
        
        thrsh(cflp,spontlp) = find_CF_Threshold_BEZ2018(CF,Fs,cohc,cihc,species,noiseType,implnt,spont,tabs,trel);
        
    end
    
    thrsh_meanHS(cflp) = mean(thrsh(cflp,:));
    thrsh_stdHS(cflp) = std(thrsh(cflp,:));
    
end

% Composite best threshold curve (CBTC) based on new curve (NBTC) from
% Miller et al. 1997 above 1 kHz and the Liberman 1978 curve (LBTC) - see
% Fig. 3 of the Miller paper.  This is a better match to the Bruce et al.
% 2003 model BTC - see Fig. 2.
thrsh_CBTC = [20.0000 -4.5000 -4.5000 -4.5000 -4.5000 6.3000 -2.9000 -4.6000 -3.1000];
CFs_CBTC = 1e3*[0.1800 0.8400 1.0000 1.2200 2.4100 4.0500 4.8000 7.2000 10.0000];

figure
semilogx(CFs/1e3,thrsh,'kx')
hold on
semilogx(CFs_CBTC/1e3,thrsh_CBTC,'k--')
ylabel('Threshold (dB SPL)')
xlabel('CF (kHz)')
xlim([0.1 20])
set(gca,'xtick',[0.1 1 10])
set(gca,'xticklabel',[0.1 1 10])

