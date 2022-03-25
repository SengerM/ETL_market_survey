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