clear all;

% numcfs = 40; % Number of CF bins from Liberman 1978
numcfs = 10; % Use a smaller number of CF bins for a quicker simulation
CFs   = logspace(log10(400),log10(15e3),numcfs);  % CF in Hz (range for Fig. 10 of Liberman 1978)

numsponts = [4 4 12];

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

% T  = 10;  % duration in seconds over which the SR is calculated
T  = 1;  % use a shorter duration for a quicker simulation
nrep = 1;               % number of stimulus repetitions (e.g., 50);

t = 0:1/Fs:T-1/Fs; % time vector

vihc = zeros(1,length(t));

thrsh = zeros(numcfs,sum(numsponts));
thrsh_meanHS = zeros(numcfs,1);
SR = zeros(numcfs,sum(numsponts));
spontval = zeros(numcfs,sum(numsponts));

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
        
        psth = model_Synapse_BEZ2018(vihc,CF,nrep,1/Fs,noiseType,implnt,spont,tabs,trel);
        
        SR(cflp,spontlp) = sum(psth)/T;
        
        spontval(cflp,spontlp) = spont;
        
        thrsh(cflp,spontlp) = find_CF_Threshold_BEZ2018(CF,Fs,cohc,cihc,species,noiseType,implnt,spont,tabs,trel);
        
    end
    
    thrsh_meanHS(cflp) = mean(thrsh(cflp,SR(cflp,:)>18));
    
end

figure
semilogx(CFs/1e3,thrsh,'kx')
hold on
semilogx(CFs/1e3,thrsh_meanHS,'r-')
ylabel('Threshold (dB SPL)')
xlabel('CF (kHz)')
xlim([0.1 20])
set(gca,'xtick',[0.1 1 10])
set(gca,'xticklabel',[0.1 1 10])

relthrsh = thrsh - repmat(thrsh_meanHS,1,sum(numsponts));

p = polyfit(log10(max(0.1,SR(SR<=18))),relthrsh(SR<=18),1);

p_allANFs = polyfit(log10(max(0.1,SR)),relthrsh,1);

figure
semilogx(max(0.1,SR(SR<=18)),relthrsh(SR<=18),'b^')
hold on
semilogx(SR(SR>18),relthrsh(SR>18),'r^')
hold on
semilogx(logspace(-1,2,100), p(1)*log10(logspace(-1,2,100))+ p(2),'b-','linewidth',2.0)
text(0.15,-5,['thrsh = ' num2str(p(1),3) '*log10(spont)+' num2str(p(2),3)])
legend('Low & medium spont fibers','High spont fibers','Fit to low & medium spont fibers')
xlabel('Adjusted Spont Rate (/s)')
ylabel('Relative Threshold (dB)')
xlim([0.1 150])
set(gca,'xtick',[0.1 1 10 100])
set(gca,'xticklabel',[0.1 1 10 100])
