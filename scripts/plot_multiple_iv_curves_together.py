# Draft script to 

from pathlib import Path
import pandas
from grafica.plotly_utils.utils import line
from devices_info import devices_info_df

MEASUREMENTS_TO_PLOT = [
	'20220328170454_MS03_IV_Curve',
	'20220328174929_MS04_IV_Curve',
	'20220328180452_MS05_IV_Curve',
	'20220328181658_MS06_IV_Curve',
	'20220328184548_MS07_IV_Curve',
	'20220328190340_MS10_IV_Curve',
	'20220328191813_MS13_IV_Curve',
	'20220405203708_MS10_1_probe_station',
	'20220405203718_MS10_2_probe_station',
	'20220405203723_MS07_1_probe_station',
	'20220405203728_MS02_2_probe_station',
	'20220405203733_MS06_2_probe_station',
	'20220405203735_MS06_3_probe_station',
	'20220405203738_MS06_4_probe_station',
	'20220405203741_MS06_5_probe_station',
	'20220405203744_MS08_1_probe_station',
	'20220405203843_MS08_2_probe_station',
	'20220405203845_MS08_3_probe_station',
	'20220405203848_MS08_4_probe_station',
	'20220405203850_MS09_1_probe_station',
	'20220405203853_MS09_2_probe_station',
	'20220405203856_MS11PIN_1_probe_station',
	'20220405203900_MS11PIN_2_probe_station',
	'20220405203902_MS11LGAD_probe_station',
]

data_df = pandas.DataFrame()
for measurement_name in MEASUREMENTS_TO_PLOT:
	try:
		df = pandas.read_feather(Path('../../measurements_data')/Path(measurement_name)/Path('iv_curve/measured_data.fd'))
		df['Instrument'] = 'board+CAEN'
	except FileNotFoundError:
		# This is for the measurements in the probe station.
		df = pandas.read_feather(Path('../../measurements_data')/Path(measurement_name)/Path('convert_probe_station_measurement_to_our_format/measured_data.fd'))
		df['Bias voltage (V)'] = df['back_side_voltage'].abs()
		df['Bias current (A)'] = df['back_side_current'].abs()
		df['device_name'] = [s for s in measurement_name.split('_') if 'MS' in s and len(s) in {4,7,8}][0]
		df['n_voltage'] = df.index
		df['Instrument'] = 'Probe station'
	df['measurement_name'] = measurement_name
	data_df = data_df.append(df,ignore_index = True)

data_df = data_df.set_index('device_name')
data_df['ID'] = devices_info_df['ID']
data_df = data_df.reset_index()

mean_df = data_df.groupby(['measurement_name','device_name','ID','n_voltage','Instrument']).agg(['mean','std'])
mean_df = mean_df.reset_index()
mean_df.columns = [' '.join(col).strip() for col in mean_df.columns.values]

fig = line(
	data_frame = mean_df,
	x = 'Bias voltage (V) mean',
	y = 'Bias current (A) mean',
	# ~ error_y = 'Bias current (A) std',
	# ~ error_y_mode = 'band',
	color = 'device_name',
	line_group = 'measurement_name',
	hover_name = 'measurement_name',
	line_dash = 'Instrument',
	# ~ log_y = True,
	grouped_legend = True,
)
fig.show()
