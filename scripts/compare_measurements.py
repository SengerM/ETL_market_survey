from pathlib import Path
import pandas
from grafica.plotly_utils.utils import line
from devices_info import devices_info_df
import measurements
import devices_info

def collect_IV_curves(measurements_names: list):
	"""Collects several IV curves together and returns a data frame with
	the data ready to plot.
	
	Parameters
	----------
	measurements_names: list of str
		A list with the names of the measurements you want to plot.
	
	Returns
	-------
	data_df: pandas.DataFrame
		The data frame used to create the plot.
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

IV_df = collect_IV_curves(
	[
		'20220429135741_MS38_IV_Curve',
		'20220506153807_MS37_IVCurve',
		'20220509172715_MS27_IVCurve_MountedInChubut',
		'20220510173615_MS29_IVCurve_MountedInChubut',
		'20220427162903_MS40PIN_IV_Curve_Extended',
		'20220427141237_MS42PIN_IVCurve',
	]
)
fig = line(
	data_frame = IV_df,
	x = 'Bias voltage (V) mean',
	y = 'Bias current (A) mean',
	color = 'device_public_alias',
	line_group = 'measurement_name',
	hover_name = 'measurement_name',
	line_dash = 'measurement_type',
	labels = {
		'device_public_alias': 'Device',
	},
	# ~ log_y = True,
)
fig.show()
# ~ PLOT_PATH = Path.home()/Path('iv_curves_all_together.html')
# ~ fig.write_html(str(PLOT_PATH), include_plotlyjs='cdn')
# ~ print(f'Plot was saved in {PLOT_PATH}')
