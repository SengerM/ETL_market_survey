# Draft script to 

from pathlib import Path
import pandas
from grafica.plotly_utils.utils import line
from devices_info import devices_info_df

MEASUREMENTS_TO_PLOT = {
	'20220328170454_MS03_IV_Curve',
	'20220328174929_MS04_IV_Curve',
	'20220328180452_MS05_IV_Curve',
	'20220328181658_MS06_IV_Curve',
	'20220328184548_MS07_IV_Curve',
	'20220328190340_MS10_IV_Curve',
	'20220328191813_MS13_IV_Curve',
}

data_df = pandas.DataFrame()
for measurement_name in MEASUREMENTS_TO_PLOT:
	df = pandas.read_feather(Path('../../measurements_data')/Path(measurement_name)/Path('iv_curve/measured_data.fd'))
	df['measurement_name'] = measurement_name
	data_df = data_df.append(df,ignore_index = True)

data_df = data_df.set_index('device_name')
data_df['ID'] = devices_info_df['ID']
data_df = data_df.reset_index()

mean_df = data_df.groupby(['measurement_name','device_name','ID','n_voltage']).agg(['mean','std'])
mean_df = mean_df.reset_index()
mean_df.columns = [' '.join(col).strip() for col in mean_df.columns.values]

fig = line(
	data_frame = mean_df,
	x = 'Bias voltage (V) mean',
	y = 'Bias current (A) mean',
	error_y = 'Bias current (A) std',
	error_y_mode = 'band',
	color = 'ID',
	line_group = 'measurement_name',
	hover_name = 'measurement_name',
	# ~ log_y = True,
)
fig.show()
