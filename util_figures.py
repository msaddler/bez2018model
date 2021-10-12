import sys
import os
import numpy as np
import matplotlib.pyplot
import matplotlib.ticker
import matplotlib.cm
import matplotlib.colors

import util_stimuli


def get_color_list(num_colors, cmap_name='Accent'):
    '''
    Helper function returns list of colors for plotting.
    '''
    if isinstance(cmap_name, list):
        return cmap_name
    cmap = matplotlib.pyplot.get_cmap(cmap_name)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=num_colors-1)
    scalar_map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    color_list = [scalar_map.to_rgba(x) for x in range(num_colors)]
    return color_list


def format_axes(ax,
                str_title=None,
                str_xlabel=None,
                str_ylabel=None,
                fontsize_title=12,
                fontsize_labels=12,
                fontsize_ticks=12,
                fontweight_title=None,
                fontweight_labels=None,
                xscale='linear',
                yscale='linear',
                xlimits=None,
                ylimits=None,
                xticks=None,
                yticks=None,
                xticks_minor=None,
                yticks_minor=None,
                xticklabels=None,
                yticklabels=None,
                spines_to_hide=[],
                major_tick_params_kwargs_update={},
                minor_tick_params_kwargs_update={}):
    '''
    Helper function for setting axes-related formatting parameters.
    '''
    ax.set_title(str_title, fontsize=fontsize_title, fontweight=fontweight_title)
    ax.set_xlabel(str_xlabel, fontsize=fontsize_labels, fontweight=fontweight_labels)
    ax.set_ylabel(str_ylabel, fontsize=fontsize_labels, fontweight=fontweight_labels)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)
    
    if xticks_minor is not None:
        ax.set_xticks(xticks_minor, minor=True)
    if yticks_minor is not None:
        ax.set_yticks(yticks_minor, minor=True)
    if xticks is not None:
        ax.set_xticks(xticks, minor=False)
    if yticks is not None:
        ax.set_yticks(yticks, minor=False)
    if xticklabels is not None:
        ax.set_xticklabels([], minor=True)
        ax.set_xticklabels(xticklabels, minor=False)
    if yticklabels is not None:
        ax.set_yticklabels([], minor=True)
        ax.set_yticklabels(yticklabels, minor=False)
    
    major_tick_params_kwargs = {
        'axis': 'both',
        'which': 'major',
        'labelsize': fontsize_ticks,
        'length': fontsize_ticks/2,
        'direction': 'out',
    }
    major_tick_params_kwargs.update(major_tick_params_kwargs_update)
    ax.tick_params(**major_tick_params_kwargs)
    
    minor_tick_params_kwargs = {
        'axis': 'both',
        'which': 'minor',
        'labelsize': fontsize_ticks,
        'length': fontsize_ticks/4,
        'direction': 'out',
    }
    minor_tick_params_kwargs.update(minor_tick_params_kwargs_update)
    ax.tick_params(**minor_tick_params_kwargs)
    
    for spine_key in spines_to_hide:
        ax.spines[spine_key].set_visible(False)
    
    return ax


def make_line_plot(ax,
                   x,
                   y,
                   legend_on=False,
                   kwargs_plot={},
                   kwargs_legend={},
                   **kwargs_format_axes):
    '''
    Helper function for basic line plot with optional legend.
    '''
    kwargs_plot_tmp = {
        'marker': '',
        'ls': '-',
        'color': [0, 0, 0],
        'lw': 1,
    }
    kwargs_plot_tmp.update(kwargs_plot)
    ax.plot(x, y, **kwargs_plot_tmp)
    ax = format_axes(ax, **kwargs_format_axes)
    if legend_on:
        kwargs_legend_tmp = {
            'loc': 'lower right',
            'frameon': False,
            'handlelength': 1.0,
            'markerscale': 1.0,
            'fontsize': 12,
        }
        kwargs_legend_tmp.update(kwargs_legend)
        ax.legend(**kwargs_legend_tmp)
    return ax


def make_nervegram_plot(ax,
                        nervegram,
                        sr=20000,
                        cfs=None,
                        cmap=matplotlib.cm.gray,
                        cbar_on=False,
                        fontsize_labels=12,
                        fontsize_ticks=12,
                        fontweight_labels=None,
                        nxticks=6,
                        nyticks=5,
                        tmin=None,
                        tmax=None,
                        treset=True,
                        vmin=None,
                        vmax=None,
                        vticks=None,
                        str_clabel=None,
                        **kwargs_format_axes):
    '''
    Helper function for visualizing auditory nervegram (or similar) representation.
    '''
    # Trim nervegram if tmin and tmax are specified
    nervegram = np.squeeze(nervegram)
    assert len(nervegram.shape) == 2, "nervegram must be freq-by-time array"
    t = np.arange(0, nervegram.shape[1]) / sr
    if (tmin is not None) and (tmax is not None):
        t_IDX = np.logical_and(t >= tmin, t < tmax)
        t = t[t_IDX]
        nervegram = nervegram[:, t_IDX]
    if treset:
        t = t - t[0]
    # Setup time and frequency ticks and labels
    time_idx = np.linspace(0, t.shape[0]-1, nxticks, dtype=int)
    time_labels = ['{:.0f}'.format(1e3 * t[itr0]) for itr0 in time_idx]
    if cfs is None:
        cfs = np.arange(0, nervegram.shape[0])
    else:
        cfs = np.array(cfs)
        assert cfs.shape[0] == nervegram.shape[0], "cfs.shape[0] must match nervegram.shape[0]"
    freq_idx = np.linspace(0, cfs.shape[0]-1, nyticks, dtype=int)
    freq_labels = ['{:.0f}'.format(cfs[itr0]) for itr0 in freq_idx]
    # Display nervegram image
    im_nervegram = ax.imshow(nervegram,
                             origin='lower',
                             aspect='auto',
                             extent=[0, nervegram.shape[1], 0, nervegram.shape[0]],
                             cmap=cmap,
                             vmin=vmin,
                             vmax=vmax)
    # Add colorbar if `cbar_on == True`
    if cbar_on:
        cbar = matplotlib.pyplot.colorbar(im_nervegram, ax=ax, pad=0.02)
        cbar.ax.set_ylabel(str_clabel, fontsize=fontsize_labels, fontweight=fontweight_labels)
        if vticks is not None:
            cbar.set_ticks(vticks)
        else:
            cbar.ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nyticks, integer=True))
        cbar.ax.tick_params(direction='out',
                            axis='both',
                            which='both',
                            labelsize=fontsize_ticks,
                            length=fontsize_ticks/2)
        cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%03d'))
    # Format axes
    ax = format_axes(ax,
                     xticks=time_idx,
                     yticks=freq_idx,
                     xticklabels=time_labels,
                     yticklabels=freq_labels,
                     fontsize_labels=fontsize_labels,
                     fontsize_ticks=fontsize_ticks,
                     fontweight_labels=fontweight_labels,
                     **kwargs_format_axes)
    return ax


def make_stimulus_summary_plot(ax_arr,
                               ax_idx_waveform=None,
                               ax_idx_spectrum=None,
                               ax_idx_nervegram=None,
                               ax_idx_excitation=None,
                               waveform=None,
                               nervegram=None,
                               sr_waveform=None,
                               sr_nervegram=None,
                               cfs=None,
                               tmin=None,
                               tmax=None,
                               treset=True,
                               vmin=None,
                               vmax=None,
                               n_anf=None,
                               erb_freq_axis=True,
                               spines_to_hide_waveform=[],
                               spines_to_hide_spectrum=[],
                               spines_to_hide_excitation=[],
                               nxticks=6,
                               nyticks=6,
                               kwargs_plot={},
                               limits_buffer=0.1,
                               ax_arr_clear_leftover=True,
                               **kwargs_format_axes):
    '''
    Helper function for generating waveform, power spectrum, nervegram, and excitation pattern
    plots to summarize a stimulus.
    '''
    # Axes are tracked in flattened array
    ax_arr = np.array([ax_arr]).reshape([-1])
    assert len(ax_arr.shape) == 1
    ax_idx_list = []
    
    # Plot stimulus waveform
    if ax_idx_waveform is not None:
        ax_idx_list.append(ax_idx_waveform)
        y_wav = np.squeeze(waveform)
        assert len(y_wav.shape) == 1, "waveform must be 1D array"
        x_wav = np.arange(0, y_wav.shape[0]) / sr_waveform
        if (tmin is not None) and (tmax is not None):
            IDX = np.logical_and(x_wav >= tmin, x_wav < tmax)
            x_wav = x_wav[IDX]
            y_wav = y_wav[IDX]
        if treset:
            x_wav = x_wav - x_wav[0]
        xlimits_wav = [x_wav[0], x_wav[-1]]
        ylimits_wav = [np.max(np.abs(y_wav)), -np.max(np.abs(y_wav))]
        ylimits_wav = np.array(ylimits_wav) * (1 + limits_buffer)
        make_line_plot(ax_arr[ax_idx_waveform],
                       x_wav,
                       y_wav,
                       legend_on=False,
                       kwargs_plot=kwargs_plot,
                       kwargs_legend={},
                       xlimits=xlimits_wav,
                       ylimits=ylimits_wav,
                       xticks=[],
                       yticks=[],
                       xticklabels=[],
                       yticklabels=[],
                       spines_to_hide=spines_to_hide_waveform,
                       **kwargs_format_axes)
    
    # Plot stimulus power spectrum
    if ax_idx_spectrum is not None:
        ax_idx_list.append(ax_idx_spectrum)
        fxx, pxx = util_stimuli.power_spectrum(waveform, sr_waveform)
        if cfs is not None:
            IDX = np.logical_and(fxx >= np.min(cfs), fxx <= np.max(cfs))
            pxx = pxx[IDX]
            fxx = fxx[IDX]
        xlimits_pxx = [np.max(pxx) * (1 + limits_buffer), 0] # Reverses x-axis                                                                                                     
        xlimits_pxx = np.ceil(np.array(xlimits_pxx) * 5) / 5
        if erb_freq_axis:
            fxx = util_stimuli.freq2erb(fxx)
            xlimits_buffer_pxx = limits_buffer * np.max(pxx)
            ylimits_fxx = [np.min(fxx), np.max(fxx)]
            yticks = np.linspace(ylimits_fxx[0], ylimits_fxx[-1], nyticks)
            yticklabels = ['{:.0f}'.format(yt) for yt in util_stimuli.erb2freq(yticks)]
        else:
            ylimits_fxx = [np.min(fxx), np.max(fxx)]
            yticks = np.linspace(ylimits_fxx[0], ylimits_fxx[-1], nyticks)
            yticklabels = ['{:.0f}'.format(yt) for yt in yticks]
        make_line_plot(ax_arr[ax_idx_spectrum],
                       pxx,
                       fxx,
                       legend_on=False,
                       kwargs_plot=kwargs_plot,
                       str_xlabel='Power\n(dB SPL)',
                       str_ylabel='Frequency (Hz)',
                       xlimits=xlimits_pxx,
                       ylimits=ylimits_fxx,
                       xticks=xlimits_pxx,
                       yticks=yticks,
                       xticklabels=xlimits_pxx.astype(int),
                       yticklabels=yticklabels,
                       spines_to_hide=spines_to_hide_spectrum,
                       **kwargs_format_axes)
    
    # Plot stimulus nervegram
    if ax_idx_nervegram is not None:
        ax_idx_list.append(ax_idx_nervegram)
        if ax_idx_spectrum is not None:
            nervegram_nxticks = nxticks
            nervegram_nyticks = 0
            nervegram_str_xlabel = 'Time\n(ms)'
            nervegram_str_ylabel = None
        else:
            nervegram_nxticks = nxticks
            nervegram_nyticks = nyticks
            nervegram_str_xlabel = 'Time (ms)'
            nervegram_str_ylabel = 'Characteristic frequency (Hz)'
        make_nervegram_plot(ax_arr[ax_idx_nervegram],
                            nervegram,
                            sr=sr_nervegram,
                            cfs=cfs,
                            nxticks=nervegram_nxticks,
                            nyticks=nervegram_nyticks,
                            tmin=tmin,
                            tmax=tmax,
                            treset=treset,
                            vmin=vmin,
                            vmax=vmax,
                            str_xlabel=nervegram_str_xlabel,
                            str_ylabel=nervegram_str_ylabel)
    
    # Plot stimulus excitation pattern
    if ax_idx_excitation is not None:
        ax_idx_list.append(ax_idx_excitation)
        if np.all(np.mod(nervegram, 1) == 0):
            # Compute mean firing rate from spike counts if all values are integers
            x_exc = np.sum(nervegram, axis=1) / (nervegram.shape[1] / sr_nervegram)
            if n_anf is not None:
                # If a number of ANFs is specified, divide firing rate by n_anf
                x_exc = x_exc / n_anf
        else:
            # Otherwise, compute mean firing rates from instantaneous firing rates
            x_exc = np.mean(nervegram, axis=1)
        xlimits_exc = [0, np.max(x_exc) * (1 + limits_buffer)]
        xlimits_exc = np.ceil(np.array(xlimits_exc)/10) * 10
        y_exc = np.arange(0, nervegram.shape[0])
        ylimits_exc = [np.min(y_exc), np.max(y_exc)]
        make_line_plot(ax_arr[ax_idx_excitation],
                       x_exc,
                       y_exc,
                       legend_on=False,
                       kwargs_plot=kwargs_plot,
                       str_xlabel='Excitation\n(spikes/s)',
                       xlimits=xlimits_exc,
                       ylimits=ylimits_exc,
                       xticks=xlimits_exc,
                       yticks=[],
                       xticklabels=xlimits_exc.astype(int),
                       yticklabels=[],
                       spines_to_hide=spines_to_hide_excitation,
                       **kwargs_format_axes)
    
    # Clear unused axes in ax_arr
    if ax_arr_clear_leftover:
        for ax_idx in range(ax_arr.shape[0]):
            if ax_idx not in ax_idx_list:
                ax_arr[ax_idx].axis('off')
    return ax_arr
