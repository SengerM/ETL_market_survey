from pathlib import Path
import pandas
from devices_info import devices_info_df
import measurements
import devices_info

def collect_IV_curves(measurements_names: list):
	"""Collects several IV curves together and returns a data frame with
	the data ready to plot.
	
	Parameters
	----------
	measurements_names: list of str
		A list with the names of the measurements you want to collect.
	
	Returns
	-------
	data_df: pandas.DataFrame
		The data frame with the data.
	"""
	iv_curve_measurement_handlers = []
	for measurement_name in measurements_names:
		handler = measurements.MeasurementHandler(measurement_name)
		if handler.measurement_type != 'IV curve':
			raise ValueError(f'Measurement {repr(measurement_name)} is of type {repr(handler.measurement_type)}, I am expecting only "IV curve".')
		iv_curve_measurement_handlers.append(handler)

	list_of_dataframes_to_concat = []
	for handler in iv_curve_measurement_handlers:
		df = handler.measurement_data
		if len(handler.measured_devices) != 1:
			raise RuntimeError(f'Measurement {handler.measurement_name} has more than one device measured, I was expecting just one...')
		df['device_name'] = handler.measured_devices[0]
		df['measurement_name'] = handler.measurement_name
		if 'probe station' in handler.measurement_type.lower():
			df['Bias voltage (V)'] = (df['back_side_voltage']**2)**.5
			df['Bias current (A)'] = (df['back_side_current']**2)**.5
			df['n_voltage'] = df.index
		df['device_ID'] = devices_info_df.loc[handler.measured_devices[0], 'ID']
		df['measurement_type'] = handler.measurement_type
		df['device_public_alias'] = devices_info.get_device_public_alias(handler.measured_devices[0])
		list_of_dataframes_to_concat.append(df)
	data_df = pandas.concat(list_of_dataframes_to_concat, ignore_index = True)

	data_df = data_df.set_index('device_name')
	data_df['device_ID'] = devices_info_df['ID']
	data_df = data_df.reset_index()

	mean_df = data_df.groupby(['measurement_name','device_name','n_voltage','measurement_type','device_public_alias']).agg(['mean','std'])
	mean_df = mean_df.reset_index()
	mean_df.columns = [' '.join(col).strip() for col in mean_df.columns.values]
	
	return mean_df

def collect_time_resolutions_vs_voltage(measurements_names: list):
	"""Collects several time resolution vs voltage curves together and 
	returns a data frame with the data ready to plot.
	
	Parameters
	----------
	measurements_names: list of str
		A list with the names of the measurements you want to collect.
	
	Returns
	-------
	data_df: pandas.DataFrame
		The data frame with the data.
	"""
	handlers = []
	for measurement_name in measurements_names:
		handler = measurements.MeasurementHandler(measurement_name)
		if handler.measurement_type != 'beta voltage scan':
			raise ValueError(f'Measurement {repr(measurement_name)} is of type {repr(handler.measurement_type)}, I am expecting only "beta voltage scan".')
		handlers.append(handler)
	
	list_of_dataframes_to_concat = []
	for handler in handlers:
		if not (measurements.MEASUREMENTS_DATA_PATH/Path(handler.measurement_name)/Path('time_resolution_vs_bias_voltage_beta_scan/script_successfully_applied')).is_file():
			raise RuntimeError(f'Measurement {handler.measurement_name} does not seem to contain "time resolution vs bias voltage" data.')
		df = pandas.read_csv(measurements.MEASUREMENTS_DATA_PATH/Path(handler.measurement_name)/Path('time_resolution_vs_bias_voltage_beta_scan/time_resolution_vs_bias_voltage.csv'))
		df['device_name'] = handler.measured_devices[0]
		df['measurement_name'] = handler.measurement_name
		df['device_ID'] = devices_info_df.loc[handler.measured_devices[0], 'ID']
		df['device_public_alias'] = devices_info.get_device_public_alias(handler.measured_devices[0])
		list_of_dataframes_to_concat.append(df)
	data_df = pandas.concat(list_of_dataframes_to_concat, ignore_index = True)
	
	return data_df

def collect_IPD_vs_voltage(measurements_names: list):
	"""Collects several inter-pixel distance vs voltage curves together and 
	returns a data frame with the data ready to plot.
	
	Parameters
	----------
	measurements_names: list of str
		A list with the names of the measurements you want to collect.
	
	Returns
	-------
	data_df: pandas.DataFrame
		The data frame with the data.
	"""
	handlers = []
	for measurement_name in measurements_names:
		handler = measurements.MeasurementHandler(measurement_name)
		if handler.measurement_type != 'TCT 1D scan sweeping bias voltage':
			raise ValueError(f'Measurement {repr(measurement_name)} is of type {repr(handler.measurement_type)}, I am expecting only "TCT 1D scan sweeping bias voltage".')
		handlers.append(handler)
	
	list_of_dataframes_to_concat = []
	for handler in handlers:
		if not (measurements.MEASUREMENTS_DATA_PATH/Path(handler.measurement_name)/Path('calculate_inter_pixel_distance_vs_bias_voltage_for_a_collection_of_1D_scans_vs_bias_voltage/script_successfully_applied')).is_file():
			raise RuntimeError(f'Measurement {handler.measurement_name} does not seem to contain "inter-pixel distance" data.')
		df = pandas.read_csv(measurements.MEASUREMENTS_DATA_PATH/Path(handler.measurement_name)/Path('calculate_inter_pixel_distance_vs_bias_voltage_for_a_collection_of_1D_scans_vs_bias_voltage/inter_pixel_distance_vs_bias_voltage.csv'))
		df['Bias voltage (V)'] = df['Bias voltage (V) median']
		df['device_name'] = handler.measured_devices[0]
		df['measurement_name'] = handler.measurement_name
		df['device_ID'] = devices_info_df.loc[handler.measured_devices[0], 'ID']
		df['device_public_alias'] = devices_info.get_device_public_alias(handler.measured_devices[0])
		list_of_dataframes_to_concat.append(df)
	data_df = pandas.concat(list_of_dataframes_to_concat, ignore_index = True)
	
	return data_df

def collect_collected_charge_vs_bias_voltage(measurements_names: list):
	"""Collects several "collected charge vs voltage" curves together and 
	returns a data frame with the data ready to plot.
	
	Parameters
	----------
	measurements_names: list of str
		A list with the names of the measurements you want to collect.
	
	Returns
	-------
	data_df: pandas.DataFrame
		The data frame with the data.
	"""
	raise NotImplementedError()

if __name__ == '__main__':
	from grafica.plotly_utils.utils import line
	
	PLOTS_SETTINGS = dict(
		color = 'device_public_alias',
		line_group = 'measurement_name',
		hover_name = 'measurement_name',
		labels = {
			'device_public_alias': 'Device',
		},
	)
	
	if False:
		IV_df = collect_IV_curves(
			[
				'20220429135741_MS38_IV_Curve',
				'20220506153807_MS37_IVCurve',
				'20220509172715_MS27_IVCurve_MountedInChubut',
				'20220510173615_MS29_IVCurve_MountedInChubut',
			]
		)
		fig = line(
			data_frame = IV_df,
			x = 'Bias voltage (V) mean',
			y = 'Bias current (A) mean',
			# ~ log_y = True,
			**PLOTS_SETTINGS,
		)
		fig.show()
	
	if False:
		time_resolution_df = collect_time_resolutions_vs_voltage(
			[
				'20220504130304_BetaScan_MS38_sweeping_bias_voltage',
				'20220505141840_BetaScan_MS37_sweeping_bias_voltage',
				'20220506173929_BetaScan_MS25_sweeping_bias_voltage',
				'20220509175539_BetaScan_MS27_sweeping_bias_voltage',
				# ~ '20220414144150_BetaScan_MS12_LGAD_sweeping_bias_voltage',
				'20220324184108_BetaScan_MS06_sweeping_bias_voltage',
				'20220329121406_BetaScan_MS04_sweeping_bias_voltage',
				'20220331121825_BetaScan_MS07_sweeping_bias_voltage',
				'20220510205309_BetaScan_MS29_sweeping_bias_voltage',
			]
		)
		fig = line(
			data_frame = time_resolution_df.sort_values(by='Bias voltage (V)'),
			x = 'Bias voltage (V)',
			y = 'Time resolution (s)',
			error_y = 'sigma from Gaussian fit (s) bootstrapped error estimation',
			error_y_mode = 'band',
			markers = 'dot',
			**PLOTS_SETTINGS,
		)
		fig.show()
	
	if True:
		IPD_df = collect_IPD_vs_voltage(
			[
				'20220404001122_MS07_sweeping_bias_voltage',
				# ~ '20220419161422_MS07_sweeping_bias_voltage',
				'20220506164227_MS37_sweeping_bias_voltage',
			]
		)
		fig = line(
			data_frame = IPD_df.sort_values(by='Bias voltage (V)'),
			x = 'Bias voltage (V)',
			y = 'Inter-pixel distance (m) calibrated',
			error_y = 'Inter-pixel distance (m) MAD_std calibrated',
			error_y_mode = 'band',
			markers = 'dot',
			**PLOTS_SETTINGS,
		)
		fig.show()
