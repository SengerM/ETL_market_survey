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
		ACCEPTED_TYPES_OF_MEASUREMENTS = {'IV curve','IV curve probe station'}
		if handler.measurement_type not in ACCEPTED_TYPES_OF_MEASUREMENTS:
			raise ValueError(f'Measurement {repr(measurement_name)} is of type {repr(handler.measurement_type)}, I am expecting one of {ACCEPTED_TYPES_OF_MEASUREMENTS}.')
		iv_curve_measurement_handlers.append(handler)

	list_of_dataframes_to_concat = []
	for handler in iv_curve_measurement_handlers:
		df = handler.measurement_data
		
		# The following is to keep only the half of the curve that is "going up" in voltage ---
		df = df.sort_values('When')
		df = df.iloc[:int(len(df.index)/2)]
		
		if len(handler.measured_devices) != 1:
			raise RuntimeError(f'Measurement {handler.measurement_name} has more than one device measured, I was expecting just one...')
		df.loc[:,'device_name'] = handler.measured_devices[0]
		df.loc[:,'measurement_name'] = handler.measurement_name
		if 'probe station' in handler.measurement_type.lower():
			df.loc[:,'Bias voltage (V)'] = (df['back_side_voltage']**2)**.5
			df.loc[:,'Bias current (A)'] = (df['back_side_current']**2)**.5
			df.loc[:,'n_voltage'] = df.index
		df.loc[:,'device_ID'] = devices_info_df.loc[handler.measured_devices[0], 'ID']
		df.loc[:,'measurement_type'] = handler.measurement_type
		df.loc[:,'device_public_alias'] = devices_info.get_device_public_alias(handler.measured_devices[0])
		df.loc[:,'Gain layer depth'] = devices_info.devices_info_df.loc[handler.measured_devices[0],'Gain layer depth']
		list_of_dataframes_to_concat.append(df)
	data_df = pandas.concat(list_of_dataframes_to_concat, ignore_index = True)

	data_df = data_df.set_index('device_name')
	data_df['device_ID'] = devices_info_df['ID']
	data_df = data_df.reset_index()

	mean_df = data_df.groupby(['measurement_name','device_name','n_voltage','measurement_type','device_public_alias','Gain layer depth']).agg(['mean','std'])
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
		df.loc[:,'device_name'] = handler.measured_devices[0]
		df.loc[:,'measurement_name'] = handler.measurement_name
		df.loc[:,'device_ID'] = devices_info_df.loc[handler.measured_devices[0], 'ID']
		df.loc[:,'device_public_alias'] = devices_info.get_device_public_alias(handler.measured_devices[0])
		df.loc[:,'Gain layer depth'] = devices_info.devices_info_df.loc[handler.measured_devices[0],'Gain layer depth']
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
		df.loc[:,'Bias voltage (V)'] = df['Bias voltage (V) median']
		df.loc[:,'device_name'] = handler.measured_devices[0]
		df.loc[:,'measurement_name'] = handler.measurement_name
		df.loc[:,'device_ID'] = devices_info_df.loc[handler.measured_devices[0], 'ID']
		df.loc[:,'device_public_alias'] = devices_info.get_device_public_alias(handler.measured_devices[0])
		df.loc[:,'Gain layer depth'] = devices_info.devices_info_df.loc[handler.measured_devices[0],'Gain layer depth']
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
	handlers = []
	for measurement_name in measurements_names:
		handler = measurements.MeasurementHandler(measurement_name)
		if handler.measurement_type != 'beta voltage scan':
			raise ValueError(f'Measurement {repr(measurement_name)} is of type {repr(handler.measurement_type)}, I am expecting only "beta voltage scan".')
		handlers.append(handler)
	
	list_of_dataframes_to_concat = []
	for handler in handlers:
		if not (measurements.MEASUREMENTS_DATA_PATH/Path(handler.measurement_name)/Path('collected_charge_vs_bias_voltage_beta_scan/script_successfully_applied')).is_file():
			raise RuntimeError(f'Measurement {handler.measurement_name} does not seem to contain "collected_charge_vs_bias_voltage_beta_scan.py" data.')
		df = pandas.read_csv(measurements.MEASUREMENTS_DATA_PATH/Path(handler.measurement_name)/Path('collected_charge_vs_bias_voltage_beta_scan/collected_charge_vs_bias_voltage.csv'))
		if len(set(df['Device name'])) != 2:
			raise RuntimeError(f'I am reading the collected charge data from measurement {handler.measurement_name} and I was expecting only two devices (DUT and reference), but instead I have found these devices: {set(df["Device name"])}. As I dont know what to do, I am rising this error.')
		found_DUT = False
		for device in set(df['Device name']):
			if device in devices_info.devices_info_df.index:
				DUT = device
				found_DUT = True
		if not found_DUT:
			raise RuntimeError(f'In data from measurement {handler.measurement_name} I cannot find which one is the DUT.')
		df = df.query(f'`Device name`=="{DUT}"')
		df.loc[:,'device_name'] = handler.measured_devices[0]
		df.loc[:,'measurement_name'] = handler.measurement_name
		df.loc[:,'device_ID'] = devices_info_df.loc[handler.measured_devices[0], 'ID']
		df.loc[:,'device_public_alias'] = devices_info.get_device_public_alias(handler.measured_devices[0])
		df.loc[:,'Gain layer depth'] = devices_info.devices_info_df.loc[handler.measured_devices[0],'Gain layer depth']
		list_of_dataframes_to_concat.append(df)
	data_df = pandas.concat(list_of_dataframes_to_concat, ignore_index = True)
	
	return data_df

if __name__ == '__main__':
	from grafica.plotly_utils.utils import line
	import plotly.express as px
	import datetime
	
	SAVE_PLOTS_IN_PATH = Path.home()/Path('cernbox/projects/ETL_market_survey/plots')
	SAVE_PLOTS_IN_PATH.mkdir(exist_ok=True)
	FIGURES_SUBTITLE = f'UZH market survey, {datetime.datetime.now():%Y-%m-%d}'
	X_PIXELS = 1920*.5
	
	PLOTS_SETTINGS = dict(
		color = 'device_public_alias',
		line_group = 'measurement_name',
		hover_name = 'measurement_name',
		line_dash = 'Gain layer depth',
		labels = {
			'device_public_alias': 'Device',
			'Inter-pixel distance (m) calibrated': 'Inter-pixel distance (m)',
			'Bias current (A) mean': 'Bias current (A)',
			'Bias voltage (V) mean': 'Bias voltage (V)',
		},
		category_orders = {
			"device_public_alias": sorted(set([devices_info.get_device_public_alias(device) for device in devices_info.devices_info_df.index if devices_info.devices_info_df.loc[device,'Manufacturer'] not in {'Micron'} and not any(_ in devices_info.get_device_public_alias(device) for _ in {'T9','nan','LGAD','PIN'})])),
		},
	)
	
	BETA_SCANS_SWEEPING_BIAS_VOLTAGE = [
		# ~ '20220504130304_BetaScan_MS38_sweeping_bias_voltage',
		# ~ '20220505141840_BetaScan_MS37_sweeping_bias_voltage',
		'20220506173929_BetaScan_MS25_sweeping_bias_voltage',
		'20220509175539_BetaScan_MS27_sweeping_bias_voltage',
		# ~ '20220414144150_BetaScan_MS12_LGAD_sweeping_bias_voltage',
		'20220324184108_BetaScan_MS06_sweeping_bias_voltage',
		'20220329121406_BetaScan_MS04_sweeping_bias_voltage',
		'20220331121825_BetaScan_MS07_sweeping_bias_voltage',
		'20220510205309_BetaScan_MS29_sweeping_bias_voltage',
		'20220512130506_BetaScan_MS31_sweeping_bias_voltage',
	]
	
	if True:
		IV_df = collect_IV_curves(
			[
				# ~ '20220419143126_MS07_IVCurve',
				'20220429135741_MS38_IV_Curve',
				'20220506153807_MS37_IVCurve',
				'20220509172715_MS27_IVCurve_MountedInChubut',
				'20220510173615_MS29_IVCurve_MountedInChubut',
				'20220512121916_MS31_IVCurve_MountedInChubut',
				'20220429150825_MS31_A_Left_probe_station',
				'20220429150823_MS38_Left_probe_station',
				'20220429150823_MS37_Left_probe_station',
				'20220429150826_MS27_A_Left_probe_station',
				'20220429150826_MS29_B_Right_probe_station',
			]
		)
		fig = line(
			title = f'IV curves<br><sup>{FIGURES_SUBTITLE}</sup>',
			data_frame = IV_df,
			x = 'Bias voltage (V) mean',
			y = 'Bias current (A) mean',
			log_y = True,
			facet_row = 'measurement_type',
			**PLOTS_SETTINGS,
		)
		fig.write_html(
			str(SAVE_PLOTS_IN_PATH/Path('IV_curves_market_survey_UZH.html')),
			include_plotlyjs = 'cdn',
		)
		for formato in {'svg','pdf'}:
			fig.write_image(
				str(SAVE_PLOTS_IN_PATH/Path(f'IV_curves_market_survey_UZH.{formato}')),
				width = int(X_PIXELS),
				height = int(X_PIXELS*.5),
			)
	
	if True:
		time_resolution_df = collect_time_resolutions_vs_voltage(BETA_SCANS_SWEEPING_BIAS_VOLTAGE)
		
		collected_charge_df = collect_collected_charge_vs_bias_voltage(BETA_SCANS_SWEEPING_BIAS_VOLTAGE)
		for df in [time_resolution_df, collected_charge_df]:
			df.set_index(['device_name','Bias voltage (V)'], inplace=True)
		for col in ['Collected charge (C)','Collected charge (C) std']:
			time_resolution_df[col] = collected_charge_df[col]
		time_resolution_df = time_resolution_df.reset_index()
		
		# ~ time_resolution_df = time_resolution_df.sample(frac=.4)
		
		fig = line(
			title = f'Time resolution<br><sup>{FIGURES_SUBTITLE}</sup>',
			data_frame = time_resolution_df.sort_values(by='Bias voltage (V)'),
			x = 'Bias voltage (V)',
			y = 'Time resolution (s)',
			# ~ error_y = 'sigma from Gaussian fit (s) bootstrapped error estimation',
			# ~ error_y_mode = 'band',
			markers = True,
			**PLOTS_SETTINGS,
		)
		fig.update_traces(
			marker = dict(
				size = 4,
			)
		)
		
		fig.write_html(
			str(SAVE_PLOTS_IN_PATH/Path('time_resolution_market_survey_UZH.html')),
			include_plotlyjs = 'cdn',
		)
	
	if True:
		collected_charge_df = collect_collected_charge_vs_bias_voltage(BETA_SCANS_SWEEPING_BIAS_VOLTAGE)
		fig = line(
			title = f'Collected charge<br><sup>{FIGURES_SUBTITLE}</sup>',
			data_frame = collected_charge_df.sort_values(by='Bias voltage (V)'),
			x = 'Bias voltage (V)',
			y = 'Collected charge (C)',
			error_y = 'Collected charge (C) std',
			error_y_mode = 'band',
			markers = 'dot',
			**PLOTS_SETTINGS,
		)
		fig.update_traces(
			marker = dict(
				size = 4,
			)
		)
		fig.write_html(
			str(SAVE_PLOTS_IN_PATH/Path('collected_charge_market_survey_UZH.html')),
			include_plotlyjs = 'cdn',
		)
	
	if True:
		IPD_df = collect_IPD_vs_voltage(
			[
				'20220404001122_MS07_sweeping_bias_voltage',
				'20220419161422_MS07_sweeping_bias_voltage',
				'20220506164227_MS37_sweeping_bias_voltage',
				'20220512131933_MS38_sweeping_bias_voltage',
			]
		)
		fig = line(
			title = f'Inter-pixel distance<br><sup>{FIGURES_SUBTITLE}</sup>',
			data_frame = IPD_df.sort_values(by='Bias voltage (V)'),
			x = 'Bias voltage (V)',
			y = 'Inter-pixel distance (m) calibrated',
			error_y = 'Inter-pixel distance (m) MAD_std calibrated',
			error_y_mode = 'band',
			markers = 'dot',
			**PLOTS_SETTINGS,
		)
		fig.update_traces(
			marker = dict(
				size = 4,
			)
		)
		fig.write_html(
			str(SAVE_PLOTS_IN_PATH/Path('inter-pixel_distance_market_survey_UZH.html')),
			include_plotlyjs = 'cdn',
		)
