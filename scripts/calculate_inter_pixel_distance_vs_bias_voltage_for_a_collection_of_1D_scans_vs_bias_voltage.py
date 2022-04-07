import pandas
from bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import warnings
from utils import read_measurement_list
import measurements
from grafica.plotly_utils.utils import line
import numpy as np

from fit_erf_and_calculate_calibration_factor import script_core as fit_erf_and_calculate_calibration_factor
from calculate_inter_pixel_distance_for_single_1D_scan import script_core as calculate_inter_pixel_distance_for_single_1D_scan

def script_core(directory: Path, force: bool=False):
	Adérito = Bureaucrat(
		directory,
		new_measurement = False,
		variables = locals(),
	)

	with Adérito.verify_no_errors_context():
		data_df = pandas.DataFrame(columns = ['Bias voltage (V) median','Bias voltage (V) MAD_std','Inter-pixel distance (m)','Distance calibration factor'])
		for measurement_name in read_measurement_list(Adérito.measurement_base_path):
			print(f"Running on {measurement_name}...")
			fit_erf_and_calculate_calibration_factor(
				Adérito.measurement_base_path.parent/Path(measurement_name),
				force = force,
				window_size = 300e-6, # From the microscope pictures.
			)
			calculate_inter_pixel_distance_for_single_1D_scan(
				Adérito.measurement_base_path.parent/Path(measurement_name),
				force = force,
				window_size = 300e-6, # From the microscope pictures.
				rough_inter_pixel_distance = 100e-6,
				number_of_bootstrap_replicas_to_estimate_uncertainty = 33,
			)
			# Here we are sure that things were calculated, so now we can put everything together ---
			measurement_handler = measurements.MeasurementHandler(measurement_name)
			
			this_measurement_stuff = {
				'Bias voltage (V) median': np.abs(measurement_handler.bias_voltage_summary['median']),
				'Bias voltage (V) MAD_std': measurement_handler.bias_voltage_summary['MAD_std'],
				'Inter-pixel distance (m)': measurement_handler.inter_pixel_distance_summary['Inter-pixel distance (m) value on data'],
				'Inter-pixel distance (m) MAD_std': measurement_handler.inter_pixel_distance_summary['Inter-pixel distance (m) MAD_std'],
				'Distance calibration factor': measurement_handler.distance_calibration_factor,
			}
			data_df = pandas.concat([data_df, pandas.DataFrame(this_measurement_stuff, index=[0])], ignore_index = True)
		data_df = data_df.sort_values(by='Bias voltage (V) median')
		for col in {'Inter-pixel distance (m)','Inter-pixel distance (m) MAD_std'}:
			data_df[f'{col} calibrated'] = data_df[col]*data_df['Distance calibration factor']
		
		for y_var in {'',' calibrated'}:
			fig = line(
				title = f'Inter pixel distance<br><sup>Measurement: {Adérito.measurement_name}</sup>',
				data_frame = data_df,
				x = 'Bias voltage (V) median',
				y = f'Inter-pixel distance (m){y_var}',
				error_y = f'Inter-pixel distance (m) MAD_std{y_var}',
				error_y_mode = 'band',
				markers = True,
				labels = {
					'Inter-pixel distance (m)': 'Inter-pixel distance (m) without calibration',
					'Inter-pixel distance (m) calibrated': 'Inter-pixel distance (m)',
					'Bias voltage (V) median': 'Bias voltage (V)',
				},
			)
			fig.write_html(
				str(Adérito.processed_data_dir_path/Path(f'inter pixel distance{y_var}.html')),
				include_plotlyjs = 'cdn',
			)

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--dir',
		metavar = 'path',
		help = 'Path to the base directory of a beta scan sweep',
		required = True,
		dest = 'directory',
		type = str,
	)
	args = parser.parse_args()
	script_core(Path(args.directory), force=False)
