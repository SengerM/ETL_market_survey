import pandas
from bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import plotly.express as px
import numpy as np
import warnings
import grafica.plotly_utils.utils as plotly_utils

from utils import read_measurement_list, get_voltage_from_measurement
from time_resolution_beta_scan_single_voltage import script_core as time_resolution_beta_scan_single_voltage

def script_core(directory: Path, force:bool=False, force_calculation_at_each_point:bool=False):
	Teutónio = Bureaucrat(
		directory,
		new_measurement = False,
		variables = locals(),
	)

	if force == False and Teutónio.job_successfully_completed_by_script('this script'):
		return

	with Teutónio.verify_no_errors_context():
		time_resolution_data = []
		for measurement_name in read_measurement_list(Teutónio.measurement_base_path):
			time_resolution_beta_scan_single_voltage(
				directory = directory.parent/Path(measurement_name),
				force = force_calculation_at_each_point,
				n_bootstrap = 22,
			)
			df = pandas.read_csv(Teutónio.measurement_base_path.parent/Path(measurement_name)/Path('time_resolution_beta_scan_single_voltage/results.csv'))
			bootstrap_df = pandas.read_csv(Teutónio.measurement_base_path.parent/Path(measurement_name)/Path('time_resolution_beta_scan_single_voltage/bootstrap_results.csv'))
			this_measurement_error = bootstrap_df['sigma from Gaussian fit (s)'].std()
			time_resolution_data.append(
				{
					'sigma from Gaussian fit (s)': float(list(df.query('type=="estimator value on the data"')['sigma from Gaussian fit (s)'])[0]),
					'sigma from Gaussian fit (s) bootstrapped error estimation': this_measurement_error,
					'Measurement name': measurement_name,
					'Bias voltage (V)': int(get_voltage_from_measurement(measurement_name)[:-1]),
				}
			)

		time_resolution_df = pandas.DataFrame.from_records(time_resolution_data)

		# ~ REFERENCE_TIME_RESOLUTION = 36.9e-12 # Speedy Gonzalez 12 Time Resolution
		time_resolution_df['Time resolution (s)'] = time_resolution_df['sigma from Gaussian fit (s)']/2**.5

		df = time_resolution_df.sort_values(by='Bias voltage (V)')
		fig = plotly_utils.line(
			title = f'Time resolution vs bias voltage with beta source<br><sup>Measurement: {Teutónio.measurement_name}</sup>',
			data_frame = df,
			x = 'Bias voltage (V)',
			y = 'Time resolution (s)',
			error_y = 'sigma from Gaussian fit (s) bootstrapped error estimation',
			hover_data = sorted(df),
			markers = 'circle',
		)
		fig.write_html(
			str(Teutónio.processed_data_dir_path/Path('time resolution vs bias voltage.html')),
			include_plotlyjs = 'cdn',
		)
		time_resolution_df.to_csv(Teutónio.processed_data_dir_path/Path('time_resolution_vs_bias_voltage.csv'))

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--dir',
		metavar = 'path',
		help = 'Path to the base directory of a measurement',
		required = True,
		dest = 'directory',
		type = str,
	)
	parser.add_argument(
		'--force-calculation-at-each-point',
		help = 'If passed, the time resolution at each point will be recalculated independently of whether it was already calculated or not before.',
		action = 'store_true',
		dest = 'force_calculation_at_each_point',
	)
	args = parser.parse_args()
	script_core(
		Path(args.directory), 
		force = True,
		force_calculation_at_each_point = args.force_calculation_at_each_point,
	)
