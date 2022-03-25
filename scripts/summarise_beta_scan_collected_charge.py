import pandas
from bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import warnings
import grafica.plotly_utils.utils as plotly_utils

from utils import read_measurement_list, get_voltage_from_measurement

def script_core(directory: Path, dut_name: str, force: bool=False):
	Vitorino = Bureaucrat(
		directory,
		new_measurement = False,
		variables = locals(),
	)

	if force == False and Vitorino.job_successfully_completed_by_script('this script'):
		return

	collected_charge_df = pandas.DataFrame()
	with Vitorino.verify_no_errors_context():
		for measurement_name in read_measurement_list(Vitorino.measurement_base_path):
			if not (Vitorino.measurement_base_path.parent/Path(measurement_name)/Path('plot_collected_charge/.script_successfully_applied')).is_file():
				warnings.warn(f"Collected Charge plotter was not successfully completed for {measurement_name}")
				continue
			try:
				df = pandas.read_csv(Vitorino.measurement_base_path.parent/Path(measurement_name)/Path('plot_collected_charge/results.csv'))
			except FileNotFoundError:
				warnings.warn(f'Cannot read data from measurement {repr(measurement_name)}')
				continue
			collected_charge_df = collected_charge_df.append(
				{
					'Collected charge (V s) x_mpv': float(df.query(f'`Device name`=="{dut_name}"').query('Variable=="Collected charge (V s) x_mpv"').query('Type=="fit to data"')['Value']),
					'Measurement name': measurement_name,
					'Bias voltage (V)': get_voltage_from_measurement(measurement_name),
					'Fluence (neq/cm^2)/1e14': 0,
				},
				ignore_index = True,
			)

		df = collected_charge_df.sort_values(by='Bias voltage (V)')
		fig = plotly_utils.line(
			title = f'Collected charge vs bias voltage with beta source<br><sup>Measurement: {Vitorino.measurement_name}</sup>',
			data_frame = df,
			x = 'Bias voltage (V)',
			y = 'Collected charge (V s) x_mpv',
			hover_data = sorted(df),
			markers = 'circle',
		)
		fig.write_html(
			str(Vitorino.processed_data_dir_path/Path('collected charge vs bias voltage.html')),
			include_plotlyjs = 'cdn',
		)
		collected_charge_df.to_csv(Vitorino.processed_data_dir_path/Path('collected_charge_vs_bias_voltage.csv'))

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--dir',
		metavar = 'path',
		help = 'Path to the base directory of a measurement. If "all", the script is applied to all linear scans.',
		required = True,
		dest = 'directory',
		type = str,
	)
	args = parser.parse_args()
	dut_name = str(input('Name of DUT? '))
	script_core(Path(args.directory), dut_name=dut_name, force=True)
