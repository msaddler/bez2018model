function h = plot_neurogram(t,CFs,neurogram,varargin)

if (nargin > 4)
    error('Too many input arguments')
elseif (nargin > 3)
    hin = varargin{1};
    
    if strncmp(get(hin,'Type'),'figu',4)
        
        figure(hin)
        h = axes;
        
    elseif strncmp(get(hin,'Type'),'axes',4)
        
        h = hin;
        
    else
        error(['Handle type of ' get(hin,'Type') ' not supported'])
    end
    
else
    
    h = axes;
    
end

axes(h)
imagesc(t,log10(CFs/1e3),neurogram)
axis xy
yticks = [0.125 0.5 2 8 16];
set(h,'ytick',log10(yticks))
set(h,'yticklabel',yticks)
ylabel('CF (kHz)')
xlabel('Time (s)')
hcb = colorbar;
set(get(hcb,'ylabel'),'string','spikes')

end

