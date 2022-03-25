import pandas
from bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import plotly.express as px
import numpy as np
import warnings
import grafica.plotly_utils.utils as plotly_utils

from utils import read_measurement_list, get_voltage_from_measurement

def script_core(directory: Path, force: bool=False):
	Teutónio = Bureaucrat(
		directory,
		new_measurement = False,
		variables = locals(),
	)

	if force == False and Teutónio.job_successfully_completed_by_script('this script'):
		return

	time_resolution_df = pandas.DataFrame()
	with Teutónio.verify_no_errors_context():
		for measurement_name in read_measurement_list(Teutónio.measurement_base_path):
			if not (Teutónio.measurement_base_path.parent/Path(measurement_name)/Path('plot_time_resolution/.script_successfully_applied')).is_file():
				warnings.warn(f"Time Resolution was not successfully completed for {measurement_name}")
				continue
			try:
				df = pandas.read_csv(Teutónio.measurement_base_path.parent/Path(measurement_name)/Path('plot_time_resolution/results.csv'))
			except FileNotFoundError:
				warnings.warn(f'Cannot read data from measurement {repr(measurement_name)}')
				continue
			try:
				bootstrap_df = pandas.read_csv(Teutónio.measurement_base_path.parent/Path(measurement_name)/Path('plot_time_resolution/bootstrap_results.csv'))
				this_measurement_error = bootstrap_df['sigma from Gaussian fit (s)'].std()
			except FileNotFoundError:
				this_measurement_error = float('NaN')
			time_resolution_df = time_resolution_df.append(
				{
					'sigma from Gaussian fit (s)': float(list(df.query('type=="estimator value on the data"')['sigma from Gaussian fit (s)'])[0]),
					'sigma from Gaussian fit (s) bootstrapped error estimation': this_measurement_error,
					'Measurement name': measurement_name,
					'Bias voltage (V)': get_voltage_from_measurement(measurement_name),
					'Fluence (neq/cm^2)/1e14': 0,
				},
				ignore_index = True,
			)

		REFERENCE_TIME_RESOLUTION = 35.8e-12 # Speedy Gonzalez 12 Time Resolution
		time_resolution_df['Time resolution (s)'] = (time_resolution_df['sigma from Gaussian fit (s)']**2 - REFERENCE_TIME_RESOLUTION**2)**.5

		df = time_resolution_df.sort_values(by='Bias voltage (V)')
		fig = plotly_utils.line(
			title = f'Collected charge vs bias voltage with beta source<br><sup>Measurement: {Teutónio.measurement_name}</sup>',
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
	args = parser.parse_args()
	script_core(Path(args.directory), force=True)
