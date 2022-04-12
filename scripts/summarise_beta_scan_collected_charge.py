import pandas
from bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import warnings
import grafica.plotly_utils.utils as plotly_utils
from scipy.stats import median_abs_deviation
from utils import read_measurement_list, get_voltage_from_measurement
from plot_collected_charge import script_core as plot_collected_charge

def script_core(directory: Path, force: bool=False):
	Vitorino = Bureaucrat(
		directory,
		new_measurement = False,
		variables = locals(),
	)

	if force == False and Vitorino.job_successfully_completed_by_script('this script'):
		return

	with Vitorino.verify_no_errors_context():
		collected_charge_data = []
		for measurement_name in read_measurement_list(Vitorino.measurement_base_path):
			try:
				plot_collected_charge(
					Vitorino.measurement_base_path.parent/Path(measurement_name),
					force = False,
					n_bootstrap = 33,
				)
				df = pandas.read_csv(Vitorino.measurement_base_path.parent/Path(measurement_name)/Path('plot_collected_charge/results.csv'))
			except FileNotFoundError as e:
				warnings.warn(f'Cannot read data from measurement {repr(measurement_name)}, reason: {e}')
				continue
			for device_name in set(df['Device name']):
				collected_charge_data.append(
					{
						'Collected charge (V s) x_mpv value_on_data': float(df.query(f'`Device name`=="{device_name}"').query('Variable=="Collected charge (V s) x_mpv"').query('Type=="fit to data"')['Value']),
						'Collected charge (V s) x_mpv mean': df.query(f'`Device name`=="{device_name}"').query('Variable=="Collected charge (V s) x_mpv"')['Value'].mean(),
						'Collected charge (V s) x_mpv std': df.query(f'`Device name`=="{device_name}"').query('Variable=="Collected charge (V s) x_mpv"')['Value'].std(),
						'Collected charge (V s) x_mpv median': df.query(f'`Device name`=="{device_name}"').query('Variable=="Collected charge (V s) x_mpv"')['Value'].median(),
						'Collected charge (V s) x_mpv MAD_std': median_abs_deviation(df.query(f'`Device name`=="{device_name}"').query('Variable=="Collected charge (V s) x_mpv"')['Value']),
						'Measurement name': measurement_name,
						'Bias voltage (V)': int(get_voltage_from_measurement(measurement_name)[:-1]),
						'Fluence (neq/cm^2)/1e14': 0,
						'Device name': device_name,
					}
				)
		collected_charge_df = pandas.DataFrame.from_records(collected_charge_data)

		df = collected_charge_df.sort_values(by='Bias voltage (V)')
		fig = plotly_utils.line(
			title = f'Collected charge vs bias voltage with beta source<br><sup>Measurement: {Vitorino.measurement_name}</sup>',
			data_frame = df,
			x = 'Bias voltage (V)',
			y = 'Collected charge (V s) x_mpv value_on_data',
			error_y = 'Collected charge (V s) x_mpv std',
			hover_data = sorted(df),
			markers = 'circle',
			labels = {
				'Collected charge (V s) x_mpv value_on_data': 'Collected charge (V s)',
			},
			color = 'Device name',
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
	script_core(Path(args.directory), force=True)
