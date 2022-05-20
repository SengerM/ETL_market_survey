import pandas
from bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import plotly.express as px
import numpy as np
import warnings
import shutil

from utils import read_measurement_list, get_voltage_from_measurement
from clean_beta_scan import script_core as clean_beta_scan
import measurements

def script_core(directory: Path, force: bool=False):
	Teutónio = Bureaucrat(
		directory,
		new_measurement = False,
		variables = locals(),
	)

	if force == False and Teutónio.job_successfully_completed_by_script('this script'):
		return

	with Teutónio.verify_no_errors_context():
		data_df = []
		for measurement_name in read_measurement_list(Teutónio.measurement_base_path):
			try:
				shutil.copyfile(
					directory/Path('cuts.csv'),
					measurements.MEASUREMENTS_DATA_PATH/Path(measurement_name)/Path('cuts.csv')
				)
			except FileNotFoundError:
				raise FileNotFoundError(f'This script expects to find a file called `cuts.csv` in {directory} specifying the cuts to be applied.')
			clean_beta_scan(
				directory = measurements.MEASUREMENTS_DATA_PATH/Path(measurement_name),
			)
			measured_df = pandas.read_feather(measurements.MEASUREMENTS_DATA_PATH/Path(measurement_name)/Path('beta_scan/measured_data.fd'))
			cleaned_df = pandas.read_feather(measurements.MEASUREMENTS_DATA_PATH/Path(measurement_name)/Path('clean_beta_scan/clean_triggers.fd'))
			for df in [measured_df, cleaned_df]:
				df.set_index('n_trigger', inplace=True)
			measured_df['accepted'] = cleaned_df['accepted']
			measured_df = measured_df.reset_index()
			measured_df['Measurement name'] = measurement_name
			data_df.append(measured_df)
		data_df = pandas.concat(data_df)
		
		FACET_ROW = 'Measurement name'
		
		df = data_df
		fig = px.histogram(
			title = f'AAAAAAAAAAAAAAAAAAAA<br><sup>Measurement: {Teutónio.measurement_name}</sup>',
			data_frame = df.sort_values(by='Bias voltage (V)'),
			x = 'Collected charge (V s)',
			color = 'accepted',
			facet_row = FACET_ROW,
			facet_row_spacing = 0.01,
			facet_col = 'device_name'
		)
		fig.update_layout(
			height = 300*len(set(df[FACET_ROW])),
		)
		fig.write_html(
			str(Teutónio.processed_data_dir_path/Path(f'collected_charge.html')),
			include_plotlyjs = 'cdn',
		)

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
