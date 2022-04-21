from pathlib import Path
import pandas
from grafica.plotly_utils.utils import line
from devices_info import devices_info_df
import measurements

iv_curve_measurement_handlers = []
for measurement_path in sorted(measurements.MEASUREMENTS_DATA_PATH.iterdir()):
	handler = measurements.MeasurementHandler(measurement_path.parts[-1])
	if 'IV curve' in handler.measurement_type:
		iv_curve_measurement_handlers.append(handler)

# ~ MEASUREMENTS_TO_PLOT = [
	# ~ '20220328170454_MS03_IV_Curve',
	# ~ '20220328174929_MS04_IV_Curve',
	# ~ '20220328180452_MS05_IV_Curve',
	# ~ '20220328181658_MS06_IV_Curve',
	# ~ '20220328184548_MS07_IV_Curve',
	# ~ '20220328190340_MS10_IV_Curve',
	# ~ '20220328191813_MS13_IV_Curve',
# ~ ]

list_of_dataframes_to_concat = []
for handler in iv_curve_measurement_handlers:
	df = handler.measurement_data
	if len(handler.measured_devices) != 1:
		raise RuntimeError(f'Measurement {handler.measurement_name} has more than one device measured, I was expecting just one...')
	df['measured_device_name'] = handler.measured_devices[0]
	df['measurement_name'] = handler.measurement_name
	if 'probe station' in handler.measurement_type.lower():
		df['Bias voltage (V)'] = (df['back_side_voltage']**2)**.5
		df['Bias current (A)'] = (df['back_side_current']**2)**.5
		df['n_voltage'] = df.index
	df['device_ID'] = devices_info_df.loc[handler.measured_devices[0], 'ID']
	df['measurement_type'] = handler.measurement_type
	list_of_dataframes_to_concat.append(df)
data_df = pandas.concat(list_of_dataframes_to_concat, ignore_index = True)

data_df = data_df.set_index('measured_device_name')
data_df['device_ID'] = devices_info_df['ID']
data_df = data_df.reset_index()

mean_df = data_df.groupby(['measurement_name','measured_device_name','device_ID','n_voltage','measurement_type']).agg(['mean','std'])
mean_df = mean_df.reset_index()
mean_df.columns = [' '.join(col).strip() for col in mean_df.columns.values]

fig = line(
	data_frame = mean_df,
	x = 'Bias voltage (V) mean',
	y = 'Bias current (A) mean',
	# ~ error_y = 'Bias current (A) std',
	# ~ error_y_mode = 'band',
	color = 'device_ID',
	line_group = 'measurement_name',
	hover_name = 'measurement_name',
	line_dash = 'measurement_type',
	log_y = True,
	grouped_legend = True,
)
fig.show()
fig.write_html(
	"LGAD_IVCurve.html",
	include_plotlyjs = 'cdn',
)
