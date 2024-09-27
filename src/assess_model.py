import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import xskillscore as xs

def compute_r2(y_pred, y_true):
    # Residual sum of squares
    ss_res = ((y_true - y_pred) ** 2).sum(skipna=True)
    # Total sum of squares
    y_true_mean = y_true.mean(skipna=True)
    ss_tot = ((y_true - y_true_mean) ** 2).sum(skipna=True)
    # R-squared
    r2 = 1 - (ss_res / ss_tot)
    return r2

def compute_season_error_metrics(da1_aligned,da2_aligned,variable,resample_freq, include_metrics):
    valid_frequencies = ['D', 'W']
    if not resample_freq in valid_frequencies:
        raise ValueError(f"Invalid resample_freq: {resample_freq}. Must be one of {valid_frequencies}")
    da1_sampled = da1_aligned.resample(time=resample_freq).mean(skipna=True)
    da2_sampled = da2_aligned.resample(time=resample_freq).mean(skipna=True)

    output_data_aligned_norain = xr.where(da1_aligned > 0, 1, 0)
    station_data_aligned_norain = xr.where(da2_aligned > 0, 1, 0)

    season_groups = np.unique(da1_aligned['time.season'])
    season_metrics = []
    for time_group in season_groups:
        output_period_data = da1_sampled.sel(time=da1_sampled['time.season']==time_group)
        station_period_data = da2_sampled.sel(time=da2_sampled['time.season']==time_group)
        valid_mask = np.logical_and(~np.isnan(output_period_data), ~np.isnan(station_period_data))

        rmse = xs.rmse(output_period_data,station_period_data, skipna=True).values.item()
        mae = xs.mae(output_period_data,station_period_data, skipna=True).values.item()
        #r2 = compute_r2(output_period_data,station_period_data).values.item()
        r2 = xs.r2(output_period_data,station_period_data, skipna=True).values.item()

        binary_accuracy = None
        if variable == 'precip':
            output_period_data = output_data_aligned_norain.sel(time=output_data_aligned_norain['time.season']==time_group)
            station_period_data = station_data_aligned_norain.sel(time=station_data_aligned_norain['time.season']==time_group)
            rain_contingency_table = xs.Contingency(output_period_data.where(valid_mask),
                                                    station_period_data.where(valid_mask),
                                                    np.array([0, 1, 999]),
                                                    np.array([0, 1, 999]),
                                                    dim=['time'])
            binary_accuracy = rain_contingency_table.accuracy().item()
        ssn_metrics = pd.DataFrame([[rmse,mae,r2,binary_accuracy]], columns=[metric+'_'+time_group for metric in ['rmse','mae','r2','binary-accuracy']])
        season_metrics.append(ssn_metrics)
    result_df = pd.concat(season_metrics, axis=1)
    result_df = result_df[[col for col in result_df.columns if col.split('_')[0] in include_metrics]]
    return result_df

def compute_error_metrics(da1_aligned,da2_aligned,variable,resample_freq):
    valid_frequencies = ['D', 'W', 'M']
    if not resample_freq in valid_frequencies:
        raise ValueError(f"Invalid resample_freq: {resample_freq}. Must be one of {valid_frequencies}")
 
    da1_sampled = da1_aligned.resample(time=resample_freq).mean(skipna=True)
    da2_sampled = da2_aligned.resample(time=resample_freq).mean(skipna=True)
    mae = xs.mae(da1_sampled,da2_sampled, skipna=True).values.item()
    rmse = xs.rmse(da1_sampled,da2_sampled, skipna=True).values.item()
    r2 = xs.r2(da1_sampled,da2_sampled, skipna=True).values.item()
    # for rain, binary accuracy
    binary_accuracy = None
    if variable == 'precip':
        output_data_aligned_norain = xr.where(da1_aligned > 0, 1, 0)
        station_data_aligned_norain = xr.where(da2_aligned > 0, 1, 0)
        if resample_freq in ['M']:
            if resample_freq == 'M':
                time_grouper = 'month'
            output_data_groups = np.unique(output_data_aligned_norain['time.' + time_grouper])
            station_data_groups = np.unique(output_data_aligned_norain['time.' + time_grouper])

            #Grouping by the resample frequency period, without resampling
            binary_accuracy = []
            for time_group in station_data_groups:
                output_period_data = output_data_aligned_norain.sel(time=output_data_aligned_norain['time.' + time_grouper]==time_group)
                station_period_data = station_data_aligned_norain.sel(time=station_data_aligned_norain['time.' + time_grouper]==time_group)
                valid_mask = np.logical_and(~np.isnan(output_period_data), ~np.isnan(station_period_data))
        
                rain_contingency_table = xs.Contingency(output_period_data.where(valid_mask),
                                                        station_period_data.where(valid_mask),
                                                        np.array([0, 1, 999]),
                                                        np.array([0, 1, 999]),
                                                        dim=['time'])
                
                binary_accuracy_value = rain_contingency_table.accuracy().item()
                binary_accuracy.append(binary_accuracy_value)
        else:
            rain_contingency_table = xs.Contingency(output_data_aligned_norain,
                                                    station_data_aligned_norain,
                                                    np.array([0, 1, 999]),
                                                    np.array([0, 1, 999]),
                                                    dim=['time'])
            binary_accuracy = rain_contingency_table.accuracy().item()
    metrics_df = pd.DataFrame([[rmse,mae,r2,binary_accuracy]], columns= ['rmse','mae','r2','binary-accuracy'])
    return metrics_df



def compare_monthly_means_line(da1, da2, city_bounds):
    # Subset the datasets
    subset_ds1 = da1.sel(lat=slice(city_bounds[2], city_bounds[3]), lon=slice(city_bounds[0], city_bounds[1]))
    subset_ds2 = da2.sel(lat=slice(city_bounds[2], city_bounds[3]), lon=slice(city_bounds[0], city_bounds[1]))
    # Calculate the monthly mean
    monthly_mean_ds1 = subset_ds1.resample(time='1M').mean()
    monthly_mean_ds2 = subset_ds2.resample(time='1M').mean()
    plt.figure(figsize=(6,5))
    plt.plot(monthly_mean_ds1['time'], monthly_mean_ds1.mean(dim=['lat', 'lon']), label='Reference')
    plt.plot(monthly_mean_ds2['time'], monthly_mean_ds2.mean(dim=['lat', 'lon']), label='Predicted')
    plt.legend()

def compare_monthly_iqr_line(da1, da2, city_bounds):
    # Subset the datasets
    subset_ds1 = da1.sel(lat=slice(city_bounds[2], city_bounds[3]), lon=slice(city_bounds[0], city_bounds[1]))
    subset_ds2 = da2.sel(lat=slice(city_bounds[2], city_bounds[3]), lon=slice(city_bounds[0], city_bounds[1]))
    # Calculate the monthly mean
    monthly_mdn_ds1 = subset_ds1.resample(time='1M').median()
    monthly_mdn_ds2 = subset_ds2.resample(time='1M').median()
    # Calculate the monthly percentiles
    monthly_p25_ds1 = subset_ds1.resample(time='1M').quantile(0.25)
    monthly_p25_ds2 = subset_ds2.resample(time='1M').quantile(0.25)
    monthly_p75_ds1 = subset_ds1.resample(time='1M').quantile(0.75)
    monthly_p75_ds2 = subset_ds2.resample(time='1M').quantile(0.75)
    plt.figure(figsize=(6,5))
    plt.plot(monthly_mdn_ds1['time'], monthly_mdn_ds1.mean(dim=['lat', 'lon']), label='Reference')
    plt.fill_between(monthly_mdn_ds1['time'],monthly_p25_ds1.mean(dim=['lat', 'lon']),monthly_p75_ds1.mean(dim=['lat', 'lon']), alpha=0.5)
    plt.plot(monthly_mdn_ds2['time'], monthly_mdn_ds2.mean(dim=['lat', 'lon']), label='Predicted')
    plt.fill_between(monthly_mdn_ds2['time'],monthly_p25_ds2.mean(dim=['lat', 'lon']),monthly_p75_ds2.mean(dim=['lat', 'lon']), alpha=0.5)
    plt.legend()

def compare_monthly_means_map(da1, da2):
    # Compute the monthly mean for each dataset
    monthly_mean_ds1 = da1.resample(time='1M').mean()
    monthly_mean_ds2 = da2.resample(time='1M').mean()
    vmin = min(monthly_mean_ds1.min().values, monthly_mean_ds2.min().values)
    vmax = max(monthly_mean_ds1.max().values, monthly_mean_ds2.max().values)

    fig, axes = plt.subplots(nrows=3, ncols=8, figsize=(24,8))
    axes = axes.flatten()

    # Iterate over the months
    for i, month in enumerate(monthly_mean_ds1.time):
        ax1 = axes[i*2]    # First axis for the month
        ax2 = axes[i*2+1]  # Second axis for the month
        # Plot the first dataset map
        monthly_mean_ds1.sel(time=month).plot(ax=ax1, vmin=vmin, vmax=vmax, cmap='viridis')
        ax1.set_title(f'{month.dt.strftime("%B %Y").item()} - Ref')
        ax1.set_xlabel('')
        ax1.set_ylabel('')
        # Plot the second dataset map
        monthly_mean_ds2.sel(time=month).plot(ax=ax2, vmin=vmin, vmax=vmax, cmap='viridis')
        ax2.set_title(f'{month.dt.strftime("%B %Y").item()} - Pred')
        ax2.set_xlabel('')
        ax2.set_ylabel('')

    # Adjust the layout and show the plot
    plt.tight_layout()
    plt.show()

def compare_monthly_iqr_line_multiple(ds_list, ds_labels, ds_colors, title, ylabel, ylim,ax):
    months_xticks_str = ['J','F','M','A','M','J','J','A','S','O','N','D']
    for ds, label, color in zip(ds_list, ds_labels, ds_colors):
        # Calculate the monthly mean
        monthly_mdn_ds = ds.groupby('time.month').median('time')
        # Calculate the monthly percentiles
        monthly_p25_ds = ds.groupby('time.month').quantile(0.25)
        monthly_p75_ds =ds.groupby('time.month').quantile(0.75)
        ax.plot(monthly_mdn_ds['month'], monthly_mdn_ds, label=label, color=color)
        ax.fill_between(monthly_mdn_ds['month'], monthly_p25_ds,monthly_p75_ds, color=color, alpha=0.35, lw=0)
        ax.set_title(title)
        ax.set_xticks(monthly_mdn_ds['month'], months_xticks_str)
        ax.set_xlim([1,12])
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)
        