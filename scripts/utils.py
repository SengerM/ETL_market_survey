from pathlib import Path

def read_measurement_list(directory: Path):
	with open(directory/Path("beta_scan_sweeping_bias_voltage/README.txt"), 'r') as iFile:
		measurement_list = [line.replace('\n', '') for line in iFile if "This measurement created automatically all the following measurements:" not in line]

	return measurement_list

def get_voltage_from_measurement(name: str):
	return name.split('_')[-1]

def resample_measured_data(measured_data_df):
	resampled_df = measured_data_df.groupby(by=['n_channel', 'n_position', 'Pad']).sample(frac=1, replace=True)
	return resampled_df

def tag_left_right_pad(data_df):
	"""Given a data_df with data from a single 1D scan of two pads of a 
	device, this function adds a new column indicating if the pad is 
	"left" or "right".
	"""
	channels = set(data_df['n_channel'])
	if len(channels) != 2:
		raise ValueError(f'`data_df` contains data concerning more than two channels. I can only tag left and right pads for two channels data.')
	left_data = data_df.loc[(data_df['n_position']<data_df['n_position'].mean())]
	right_data = data_df.loc[(data_df['n_position']>data_df['n_position'].mean())]
	for channel in channels:
		if left_data.loc[left_data['n_channel']==channel, 'Collected charge (V s)'].mean(skipna=True) > left_data.loc[~(left_data['n_channel']==channel), 'Collected charge (V s)'].mean(skipna=True):
			mapping = {channel: 'left', list(channels-{channel})[0]: 'right'}
		else:
			mapping = {channel: 'right', list(channels-{channel})[0]: 'left'}
	pad_df = pandas.DataFrame(index=data_df.index)
	for n_channel in set(data_df['n_channel']):
		pad_df.loc[data_df['n_channel']==n_channel, 'Pad'] = mapping[n_channel]
	return pad_df

def calculate_normalized_collected_charge(df, window_size=125e-6, laser_sigma=9e-6):
	"""df must be the dataframe from a single 1D scan. `window_size` and 
	`laser_sigma` are used to know where we expect zero signal and where 
	we expect full signal.
	Return a single-column-dataframe containint the value of the 
	normalized collected charge at each row.
	"""
	check_df_is_from_single_1D_scan(df)
	normalized_charge_df = pandas.DataFrame(index=df.index)
	normalized_charge_df['Normalized collected charge'] = df['Collected charge (V s)'].copy()
	if 'Pad' not in df.columns:
		raise RuntimeError(f'Before calling this function you have to call `tag_left_right_pad` function on your data frame.')
	if 'Distance (m)' not in df.columns:
		raise RuntimeError(f'Before calling this function you have to call `append_distance_column` function on your data frame.')
	for n_pulse in sorted(set(df['n_pulse'])):
		for pad in {'left','right'}:
			rows_where_I_expect_no_signal_i_e_where_there_is_metal = (df['Distance (m)'] < df['Distance (m)'].median() - window_size - 2*laser_sigma) | (df['Distance (m)'] > df['Distance (m)'].median() + window_size + 2*laser_sigma)
			if pad == 'left':
				rows_where_I_expect_full_signal_i_e_where_there_is_silicon = (df['Distance (m)'] > df['Distance (m)'].median() - window_size + 2*laser_sigma) & (df['Distance (m)'] < df['Distance (m)'].median() - 2*laser_sigma)
			elif pad == 'right':
				rows_where_I_expect_full_signal_i_e_where_there_is_silicon = (df['Distance (m)'] < df['Distance (m)'].median() + window_size - 2*laser_sigma) & (df['Distance (m)'] > df['Distance (m)'].median() + 2*laser_sigma)
			offset_to_subtract = normalized_charge_df.loc[rows_where_I_expect_no_signal_i_e_where_there_is_metal&(df['Pad']==pad)&(df['n_pulse']==n_pulse),'Normalized collected charge'].median()
			normalized_charge_df.loc[(df['Pad']==pad)&(df['n_pulse']==n_pulse),'Normalized collected charge'] -= offset_to_subtract
			scale_factor = normalized_charge_df.loc[rows_where_I_expect_full_signal_i_e_where_there_is_silicon&(df['Pad']==pad)&(df['n_pulse']==n_pulse),'Normalized collected charge'].median()
			normalized_charge_df.loc[(df['Pad']==pad)&(df['n_pulse']==n_pulse),'Normalized collected charge'] /= scale_factor
	return normalized_charge_df
