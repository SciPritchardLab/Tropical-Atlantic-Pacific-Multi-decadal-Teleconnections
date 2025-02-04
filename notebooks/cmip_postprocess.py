import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import scipy.stats as stats
import cartopy.crs as ccrs
import matplotlib.patches as mpatches


##### Read data and compute anomalies #####

# -- Detrending
def detrend_dim(da, dim='time', deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

def ano_norm_t(ds, rolling_time=5):
    
    idx_complete_years = ds.time.shape[0]//12
    
    clim = ds[:idx_complete_years*12].groupby('time.month').mean('time')
    
    ano = ds.groupby('time.month') - clim
    
    if rolling_time:
        ano = ano.rolling(time=rolling_time, center=True, min_periods=1).mean()
        
    ano_std = ano/ano.std()
    
  
    ano.attrs['clim_str'] = ds.time[0].values
    ano.attrs['clim_end'] = ds.time[:idx_complete_years*12][-1].values
    ano_std.attrs['clim_str'] = ds.time[0].values
    ano_std.attrs['clim_end'] = ds.time[:idx_complete_years*12][-1].values
    return ano, ano_std



def extract_region(sst, lon_bounds, lat_bounds):
    """Extract the SST data for a specific region and compute its weighted mean."""
    
    mask_lon = (sst['lon'] >= lon_bounds[0]) & (sst['lon'] <= lon_bounds[1])
    mask_lat = (sst['lat'] >= lat_bounds[0]) & (sst['lat'] <= lat_bounds[1])
    
    region = sst.where(mask_lon & mask_lat, drop=True)
    return region.weighted(np.cos(np.deg2rad(region.lat))).mean(('lon', 'lat'))
    
def compute_anomalies(sst_region, sst_index, rolling_time):
    """Detrend the SST data for a region and compute its anomalies."""
    sst_dtd = detrend_dim(sst_region, dim='time')
    ssta, ssta_norm = ano_norm_t(sst_dtd, sst_index, rolling_time)
    return ssta, ssta_norm

def read_data(path_data, data_type=2, model=None):
    import cftime
    

    ds = xr.open_dataset(path_data)

    if data_type == 1: # Hadley SST 
        ds = ds.rename({"latitude": 'lat', "longitude": 'lon'})
        sst = ds.sst.sel(time=slice('1950-01-01', '2014-12-30'))
        sst = sst.where(sst != -1000, np.nan)
    
    elif data_type == 2: # Simulations
        if isinstance(ds['time'].values[0], cftime.datetime):
            ds['time'] = np.array([np.datetime64(cftime_obj.strftime('%Y-%m-%d')) for cftime_obj in ds['time'].values])
            
        if model in ['E3SM-MMF', 'E3SMv2']:
            sst = ds.TS.sel(time=slice('1950-01-01', '2014-12-30'))
        else:
            sst = ds.ts.sel(time=slice('1950-01-01', '2014-12-30'))
       
        # sst = xr.concat([sst[:, :, 180:], sst[:, :, :180]], dim='lon')
        # sst.coords['lon'] = (sst.coords['lon'] + 180) % 360 - 180
        
        dmask = xr.open_dataset('../data/raw/lsmask.nc')
        sst = sst.where(dmask.mask.isel(time=0) == 1).squeeze()
      
    elif data_type==3: # ERSST
        sst = ds.sst.sel(time=slice('1950-01-01', '2015-12-01'))
        sst = sst.where(sst != -1000, np.nan).squeeze()
    else:
        raise ValueError("Invalid data_type. Please use 'hadley' or 'simulations'.")

    return sst 

def read_data_compute_anomalies(path_data, regions, data_type=2, rolling_time=None, sst_index='Nino', model=None):
    import cftime
    

    ds = xr.open_dataset(path_data)

    if data_type == 1: # Hadley SST 
        ds = ds.rename({"latitude": 'lat', "longitude": 'lon'})
        sst = ds.sst.sel(time=slice('1950-01-01', '2014-12-30'))
        sst = sst.where(sst != -1000, np.nan)
    
    elif data_type == 2: # Simulations
        if isinstance(ds['time'].values[0], cftime.datetime):
            ds['time'] = np.array([np.datetime64(cftime_obj.strftime('%Y-%m-%d')) for cftime_obj in ds['time'].values])
            
        if model in ['E3SM-MMF', 'E3SMv2']:
            sst = ds.TS.sel(time=slice('1950-01-01', '2014-12-30'))
        else:
            sst = ds.ts.sel(time=slice('1950-01-01', '2014-12-30'))

        
        dmask = xr.open_dataset('../data/raw/lsmask.nc')
        sst = sst.where(dmask.mask.isel(time=0) == 1).squeeze()
      
    elif data_type==3: # ERSST
        sst = ds.sst.sel(time=slice('1950-01-01', '2015-12-01'))
        sst = sst.where(sst != -1000, np.nan).squeeze()
    else:
        raise ValueError("Invalid data_type. Please use 'hadley' or 'simulations'.")

    
    ssta = {} 
    ssta_norms = {}
    for name, (lon_bounds, lat_bounds) in regions.items():
        sst_region = extract_region(sst, lon_bounds, lat_bounds)
        ssta[name], ssta_norms[name] = compute_anomalies(sst_region, sst_index, rolling_time)

    # return ssta, ssta_norms, iod_index
    return ssta, ssta_norms


##### Calculate regression ##### 

def calculate_regression_vectorize(amo_index, cloud_data, coord_name='time', sim=True):
    # Align amo_index and cloud_data along the 'time' dimension
    # if coord_name == 'time':
    #     if sim: # If data type is simulation, we need to convert the cftime to datetime 
    #         cftime_dates = [cftime.datetime(d.year, d.month, d.day) for d in cloud_data.time.values]
    #         amo_index['time'] = cftime_dates
    #         cloud_data['time'] = cftime_dates
    
    
    # non_nan_indices = np.argwhere(~np.isnan(amo_index.values)).flatten()
    # amo_index = amo_index[non_nan_indices]
    # cloud_data = cloud_data.isel({coord_name:non_nan_indices})

    amo_index['time'] = cloud_data['time']
    amo_index, cloud_data = xr.align(amo_index, cloud_data, join='inner', copy=False)

    # Define a linear regression function to apply
    def linear_regression(x, y):
        # This expects x and y to be 1D arrays.
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        return slope, intercept, r_value, p_value, std_err
    
    # Apply the linear regression function over lat and lon dimensions
    regression_results = xr.apply_ufunc(
        linear_regression,
        amo_index,                 # First input array (1D)
        cloud_data,                # Second input array (3D, will be reduced to 1D along 'time')
        input_core_dims=[[coord_name], [coord_name]],  # Core dimensions to operate over
        output_core_dims=[[], [], [], [], []],         # Output core dimensions (all scalars, so empty lists)
        vectorize=True,            # Vectorize the function over lat/lon grids
        # dask='parallelized',       # Enable dask for parallelized computation
        output_dtypes=[float, float, float, float, float]  # Define the data types of the output
    )
    
    # Assign results to variables with meaningful names
    regression_ds = xr.Dataset({
        'slope': regression_results[0],
        'intercept': regression_results[1],
        'r_value': regression_results[2],  # Note that we are returning r-squared, change if necessary
        'p_value': regression_results[3],
        'std_err': regression_results[4],
    })
    
    # Optionally, you can assign names to dimensions and variables (attributes) here
    
    return regression_ds
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
def plot_all_reg_cor_map(regression_results, var, titles, suptitle='Regression Maps for MMF and E3SM Cloud Data with AMO Index', label='Regression Coefficient',vmin=-0.1, vmax=0.1, nrows=3,ncols=3, savefig=True, outpath='/global/homes/y/yanxia/ENSO-CLOUD/Results/Regression/'):
    
    """
    Plot a series of regression correlation maps using xarray datasets.

    Parameters:
    regression_results (list of xarray.Dataset): A list of xarray Datasets containing the regression results,
                                                 each dataset in the list should have 'slope' and 'p_value' data variables.
    var (str): The variable name to plot from the datasets (typically 'slope' or 'p_value').
    titles (list of str): Titles for each subplot. The length of this list should match the length of 'regression_results'.
    suptitle (str, optional): The main title for the entire figure. Default is a preset title.
    label (str, optional): Label for the colorbar. Default is 'Regression Coefficient'.
    vmin (float, optional): The minimum value for the colormap. Default is -0.1.
    vmax (float, optional): The maximum value for the colormap. Default is 0.1.
    nrows (int, optional): The number of rows for the subplot grid. Default is 3.
    ncols (int, optional): The number of columns for the subplot grid. Default is 3.
    savefig (bool, optional): Whether to save the figure as a file. Default is True.
    outpath (str, optional): Path to save the figure if 'savefig' is True. Default is a specified directory.

    This function creates a series of maps showing regression results for different datasets.
    Each map is plotted on a subplot arranged in a grid defined by 'nrows' and 'ncols'.
    The function uses 'plot_regression_correlation_map' for individual subplots, which needs to be defined elsewhere.
    The last subplot's image object ('im') is used to create a shared colorbar for all subplots.
    If 'savefig' is True, the figure is saved to the specified 'outpath'.

    Returns:
    None
    """
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 6),
                            subplot_kw={'projection': ccrs.Robinson(central_longitude=180)}, dpi=300)
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    # Variable to hold the last 'im' object for colorbar creation
    last_im = None

    # Loop over axes and regression results to create each subplot
    for ax, ds, title in zip(axes, regression_results, titles):
        # 'ds' is assumed to be an xarray.Dataset with 'slope' and 'p_value' data variables
        last_im = plot_regression_correlation_map(ax, ds, var, title, vmin=vmin, vmax=vmax)

    # ... (the rest of the plotting code including suptitle, tight_layout, colorbar, etc.)
    # Add an overall title
    plt.suptitle(suptitle)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect so the suptitle fits

    # Add a colorbar at the bottom of the subplots
    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), orientation='horizontal', 
                        fraction=0.025, pad=0.1, aspect=50)
    cbar.set_label(label)
    if savefig:
        plt.savefig(outpath+'correlation_map.png')
    plt.show()




########## Load preprocessed data ##################
import pickle

def load_from_pickle(file_path):
    with open(file_path, 'rb') as file:
        data_dict = pickle.load(file)
    return data_dict

def save_datasets_to_pickle(dataset_dict, file_path):
    """
    Save a dictionary of xarray Datasets to a pickle file.

    Parameters:
    dataset_dict (dict): A dictionary where keys are strings and values are xarray Datasets.
    file_path (str): The path to the file where the dictionary should be saved.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(dataset_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

