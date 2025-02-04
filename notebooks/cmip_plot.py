import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
# Regresion #


################## Plot ###########################
def plot_regression_correlation_map(ax, data, variable, title, vmin=-0.1, vmax=0.1):
    # Extract the data to be plotted
    data_to_plot = data[variable]
    
    # Define a significance level
    alpha = 0.1  # Consider making this more stringent if you want fewer dots
    
    # Create a mask for significant values
    significant_mask = data['p_value'] < alpha
    
    # Plot the data
    im = ax.pcolormesh(data_to_plot['lon'], data_to_plot['lat'], data_to_plot,
                       transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=vmin, vmax=vmax)
    
    # Add stippling where the values are significant
    stipple_lons, stipple_lats = np.meshgrid(data_to_plot['lon'], data_to_plot['lat'])
    
    # Spatial thinning of significant points (e.g., every 5th point)
    thinning_factor = 5
    thin_mask = (significant_mask[::thinning_factor, ::thinning_factor])
    thin_stipple_lons = stipple_lons[::thinning_factor, ::thinning_factor]
    thin_stipple_lats = stipple_lats[::thinning_factor, ::thinning_factor]
    
    ax.scatter(thin_stipple_lons[thin_mask], thin_stipple_lats[thin_mask], 
               color='k', marker='.', s=3, alpha=0.5, transform=ccrs.PlateCarree())  # Adjust size `s` and transparency `alpha` as needed
    
    # Add coastlines for context
    ax.coastlines()
    
    # Set title for the subplot
    ax.set_title(title)
    
    # Add gridlines and labels
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.4, linestyle='--')
    gl.top_labels = False  # Disable labels at the top
    gl.right_labels = False  # Disable labels on the right
    
    # Draw a box for 'cti'
    cti_box = mpatches.Rectangle((180, -6), 90, 12, fill=False, edgecolor='black', linewidth=2, transform=ccrs.PlateCarree())
    ax.add_patch(cti_box)
    
    return im


# Create figure and axes objects for the subplots
# def plot_all_reg_cor_map(regression_results, var, titles, suptitle='Regression Maps for MMF and E3SM Cloud Data with AMO Index', \
#                          label='Regression Coefficient',vmin=-0.1, vmax=0.1, nrows=3,ncols=3, savefig=True, \
#                          outpath='/global/homes/y/yanxia/ENSO-CLOUD/Results/Regression/', out_name=None, figsize=(18,6)):
#     fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
#                             subplot_kw={'projection': ccrs.Robinson(central_longitude=180)}, dpi=300)
#     axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

#     # Variable to hold the last 'im' object for colorbar creation
#     last_im = None

#     # Loop over axes and regression results to create each subplot
#     for ax, ds, title in zip(axes, regression_results, titles):
#         # 'ds' is assumed to be an xarray.Dataset with 'slope' and 'p_value' data variables
#         last_im = plot_regression_correlation_map(ax, ds, var, title, vmin=vmin, vmax=vmax)

#     # ... (the rest of the plotting code including suptitle, tight_layout, colorbar, etc.)
#     # Add an overall title
#     plt.suptitle(suptitle)

#     # Adjust layout
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect so the suptitle fits

#     # Add a colorbar at the bottom of the subplots
#     cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), orientation='horizontal', 
#                         fraction=0.025, pad=0.1, aspect=50)
#     cbar.set_label(label)
    
#     # If there are any remaining axes not used (when n_plots is not a multiple of ncols), hide them
#     for i in range(n_plots, nrows * ncols):
#         fig.delaxes(axes[i])
        
#     if savefig:
#         plt.savefig(outpath+out_name)
#     plt.show()

    
def plot_all_reg_cor_map(regression_results, var, titles, suptitle='Regression Maps for MMF and E3SM Cloud Data with AMO Index', \
                         label='Regression Coefficient',vmin=-0.1, vmax=0.1, nrows=3, ncols=3, savefig=True, \
                         outpath='/global/homes/y/yanxia/ENSO-CLOUD/Results/Regression/', out_name='regression_map.png', figsize=(18,6)):
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                            subplot_kw={'projection': ccrs.Robinson(central_longitude=180)}, dpi=300)
    
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration
    n_plots = len(regression_results)  # Determine the total number of plots
    
    # Loop over axes and regression results to create each subplot
    for i, (ax, ds, title) in enumerate(zip(axes, regression_results, titles)):
        # Plot regression correlation map on the current axis
        last_im = plot_regression_correlation_map(ax, ds, var, title, vmin=vmin, vmax=vmax)
        
        # Set title for the subplot
        ax.set_title(title)

    # Add an overall title
    plt.suptitle(suptitle)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect so the suptitle fits

    # Add a colorbar at the bottom of the subplots
    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), orientation='horizontal', 
                        fraction=0.025, pad=0.1, aspect=50)
    cbar.set_label(label)
    
    # If there are any remaining axes not used (when n_plots is not a multiple of ncols), hide them
    for i in range(n_plots, nrows * ncols):
        fig.delaxes(axes[i])
    
    # Save the figure if savefig is True
    if savefig:
        plt.savefig(outpath + out_name)
        
    plt.show()
    
# Time series Index #

# Define the plotting function
def plot_nat_ts(data, ax, left_title=None, center_title='NAT', right_title='ROLLING 61 MONTHS'):
    ax.axhline(0, color='grey')
    ax.axhline(0.5, color='grey', linestyle='--', linewidth=2)
    ax.axhline(-0.5, color='grey', linestyle='--', linewidth=2)
    ax.plot(data.time.values, data, color='black', linewidth=2)

    # Set x-axis formatting for years
    years = mdates.YearLocator(5)   # every 5 years
    years_minor = mdates.YearLocator(1)  # every year
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_minor_locator(years_minor)
    myFmt = mdates.DateFormatter('%Y')
    ax.xaxis.set_major_formatter(myFmt)
    ax.tick_params(labelsize=20)

    ax.fill_between(data.time.values, data, 0, data > 0, color='red')
    ax.fill_between(data.time.values, data, 0, data < 0, color='blue')

    ax.set_title(left_title, fontsize=30, loc='left')
    ax.set_title(center_title, fontsize=30, loc='center')
    ax.set_title(right_title, fontsize=30, loc='right')
    
    if center_title=='CTI':
        ax.set_ylim([-4, 4])
    else:
        ax.set_ylim([-0.5, 0.5])


    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    
    
    ax.grid(True)
    # Increase the linewidth for plot edges
    for spine in ax.spines.values():
        spine.set_linewidth(4)
        
# Determine the number of subplots needed
def plot_cmip_ts_index(ssta_ds_dict, save_name=True, ncols=3, index=None, rolling_time=61):
    
    n_plots = len(ssta_ds_dict)
    ncols = ncols  # for example, you want 2 columns
    nrows = n_plots // ncols + (n_plots % ncols > 0)  # this will calculate the required

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(15 * ncols, 5 * nrows))
    axes = axes.flatten()

    for ax, (title, ds) in zip(axes, ssta_ds_dict.items()):
        plot_nat_ts(ds[index], ax, left_title=title.split('.')[0], 
                    center_title=str.upper(index),
                    right_title=f'ROLLING {rolling_time} MONTHS')

    # If there are any remaining axes not used (when n_plots is not a multiple of ncols), hide them
    for i in range(n_plots, nrows * ncols):
        fig.delaxes(axes[i])

    plt.tight_layout()

    if save_name:
        plt.savefig(f'/global/homes/y/yanxia/ENSO-CLOUD/CMIP/figures/cmip_{index}_rolling_{rolling_time}_months.png',dpi=300)

# Plot the correlation coefficient for lead and lag months
def plot_correlation_coefficients(results, title, filename, shift_time=12, x_label='(CTI Shifted) Lead/Lag Months', left_text='Pacific -> Atlantic', right_text='Atlantic -> Pacific') :
    # Inside this plot, please add a column line at x=0
    # Add text for x<0 panal as 'Pacific -> Atlantic' and x>0 panal as 'Atlantic -> Pacific' inside of plot
    from itertools import cycle
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = cycle(sns.color_palette("tab10"))
    linestyles = cycle(['-', '--', '-.', ':']) 
    for model in results.keys():
        if model in ['HadISST', 'ERSST', 'E3SM-MMF', 'E3SMv2', 'CMIP6_MMM']:
            ax.plot(range(-shift_time, shift_time+1), list(results[model][0].values()), label=model.split('.')[0], color=next(colors), linewidth=3)
        else:
            ax.plot(range(-shift_time, shift_time+1), list(results[model][0].values()), label=model.split('.')[0], color=next(colors), linestyle=next(linestyles), alpha=0.5)
    
    
    # plot the std with shaded area
    ax.fill_between(range(-shift_time, shift_time+1), 
    np.array(list(results['CMIP6_MMM'][0].values()) - np.array(list(results['CMIP6_MMM'][2].values()))), 
    np.array(list(results['CMIP6_MMM'][0].values()) + np.array(list(results['CMIP6_MMM'][2].values()))), 
    color='gray', 
    alpha=0.5)
            
    ax.axhline(0, color='black', linewidth=1, linestyle='--')
    ax.axvline(0, color='black', linewidth=1, linestyle='--')

    # ax.text(0.25, 0.9, f'{title}.split('-')[1].split(' ')[0]} → {title}.split('-')[0]',  transform=ax.transAxes, 
    #     va='center', ha='right', fontsize=14, fontweight='bold')
    # ax.text(0.6, 0.05, f'{title}.split('-')[0] → {title}.split('-')[1].split(' ')[0]', transform=ax.transAxes, 
    #     va='center', ha='left', fontsize=14, fontweight='bold')
    ax.text(0.6, 0.05, left_text, transform=ax.transAxes, 
            va='center', ha='left', fontsize=14, fontweight='bold')
    ax.text(0.25, 0.9, right_text,  transform=ax.transAxes, 
            va='center', ha='right', fontsize=14, fontweight='bold')
   
    # set the plot ticks
    interval = int(shift_time / 6)
    ax.set_xticks(range(-shift_time, shift_time + interval, interval))
    ax.set_xticklabels(range(-shift_time, shift_time + interval, interval))

    # set the plot label thickness
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # set the plot label size
    ax.tick_params(axis='both', which='major', labelsize=12, width=2)

    # set the x-axis and y-axis tick font size
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)

    # set the font style as times new roman
    # plt.rcParams['font.family'] = 'Times New Roman'

    # set the x-axis and y-axis ticks font size
    plt.xticks(fontsize=14)
    plt.xticks(fontsize=14)

    ax.set_xlabel(x_label, fontdict={'fontsize': 14}, fontweight='bold')
    ax.set_ylabel('Correlation Coefficients', fontdict={'fontsize': 14}, fontweight='bold')
    ax.set_title(title, fontdict={'fontsize': 16}, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(filename)
    plt.grid()
    # plt.close()
    plt.show()