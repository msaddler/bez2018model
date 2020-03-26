% Check to see if running under Matlab or Octave
if exist ('OCTAVE_VERSION', 'builtin') ~= 0
  pkg load signal;
  if exist('rms')<1
    rms = @(x) sqrt(mean(x.^2));
  end
end

% Set audiogram
ag_fs = [125 250 500 1e3 2e3 4e3 8e3];
ag_dbloss = [0 0 0 0 0 0 0]; % Normal hearing

% Set audiogram
% ag_fs = [125 250 500 1e3 2e3 4e3 8e3];
% ag_dbloss = [0 0 0 20 40 60 80]; % Example high-freq hearing loss

species = 2; % Human cochlear tuning (Shera et al., 2002)

[stim, Fs_stim] = audioread('defineit.wav');

stimdb = 65; % speech level in dB SPL

stim = stim/rms(stim)*20e-6*10^(stimdb/20);

[neurogram_ft,neurogram_mr,neurogram_Sout,t_ft,t_mr,t_Sout,CFs] = generate_neurogram_BEZ2018(stim,Fs_stim,species,ag_fs,ag_dbloss);

ng1=figure;
set(ng1,'renderer','painters');
winlen = 256; % Window length for the spectrogram analyses
sp1 = subplot(2,1,1);
[s,f,t] = specgram([stim; eps*ones(round(t_mr(end)*Fs_stim)-length(stim),1)],winlen,Fs_stim,winlen,0.25*winlen);
imagesc(t,f/1e3,20*log10(abs(s)/sum(hanning(winlen))*sqrt(2)/20e-6));
axis xy; axis tight;
hcb = colorbar;
set(get(hcb,'ylabel'),'string','SPL')
caxis([stimdb-80 stimdb])
ylim([0 min([max(CFs/1e3) Fs_stim/2e3])])
xlabel('Time');
ylabel('Frequency (kHz)');
title('Spectrogram')
xl = xlim;
sp2=subplot(2,1,2);
plot_neurogram(t_mr,CFs,neurogram_mr,sp2);
caxis([0 80])
title('Mean-rate Neurogram')
xlim(xl)

ng2=figure;
set(ng2,'renderer','painters');
winlen = 256; % Window length for the spectrogram analyses
sp1 = subplot(2,1,1);
[s,f,t] = specgram([stim; eps*ones(round(t_mr(end)*Fs_stim)-length(stim),1)],winlen,Fs_stim,winlen,0.25*winlen);
imagesc(t,f/1e3,20*log10(abs(s)/sum(hanning(winlen))*sqrt(2)/20e-6));
axis xy; axis tight;
hcb = colorbar;
set(get(hcb,'ylabel'),'string','SPL')
caxis([stimdb-80 stimdb])
ylim([0 min([max(CFs/1e3) Fs_stim/2e3])])
xlabel('Time');
ylabel('Frequency (kHz)');
title('Spectrogram')
xl = xlim;
sp2=subplot(2,1,2);
plot_neurogram(t_ft,CFs,neurogram_ft,sp2);
caxis([0 20])
title('Fine-timing Neurogram')
xlim(xl)

ng3=figure;
set(ng3,'renderer','painters');
winlen = 256; % Window length for the spectrogram analyses
sp1 = subplot(2,1,1);
[s,f,t] = specgram([stim; eps*ones(round(t_mr(end)*Fs_stim)-length(stim),1)],winlen,Fs_stim,winlen,0.25*winlen);
imagesc(t,f/1e3,20*log10(abs(s)/sum(hanning(winlen))*sqrt(2)/20e-6));
axis xy; axis tight;
hcb = colorbar;
set(get(hcb,'ylabel'),'string','SPL')
caxis([stimdb-80 stimdb])
ylim([0 min([max(CFs/1e3) Fs_stim/2e3])])
xlabel('Time');
ylabel('Frequency (kHz)');
title('Spectrogram')
xl = xlim;
sp2=subplot(2,1,2);
plot_neurogram(t_Sout,CFs,neurogram_Sout*diff(t_Sout(1:2)),sp2);
caxis([0 6])
title('S_{out} Neurogram')
xlim(xl)
