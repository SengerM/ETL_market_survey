import pandas
from bureaucrat.Bureaucrat import Bureaucrat # https://github.com/SengerM/bureaucrat
from pathlib import Path
import warnings
import grafica.plotly_utils.utils as plotly_utils # https://github.com/SengerM/grafica
import numpy
from scipy.stats import median_abs_deviation
from utils import read_measurement_list, get_voltage_from_measurement
from collected_charge_beta_scan_single_voltage import script_core as collected_charge_beta_scan_single_voltage
from create_calibration_for_Coulomb_conversion_in_beta_setup import read_calibration_factor
import measurements

def script_core(directory:Path, Coulomb_calibration:Path=None, force:bool=False):
	"""Plot the collected charge as a function of the bias voltage
	for a "beta scan sweeping bias voltage" measurement.
	
	Parameters
	----------
	directory: Path
		Path to the base directory of a beta scan sweeping voltage measurement.
	Coulomb_calibration: Path
		Path to a Coulomb calibration measurement type, produced by the
		script `create_calibration_for_Coulomb_conversion_in_beta_setup.py`.
	"""
	Vitorino = Bureaucrat(
		directory,
		new_measurement = False,
		variables = locals(),
	)
	
	if measurements.measurement_type(Vitorino.measurement_name) != 'beta voltage scan':
		raise NotImplementedError(f'Dont know how to process a measurement of type {repr(measurements.measurement_type(Vitorino.measurement_name))}.')

	if force == False and Vitorino.job_successfully_completed_by_script('this script'):
		return
	
	with Vitorino.verify_no_errors_context():
		collected_charge_data = []
		for measurement_name in read_measurement_list(Vitorino.measurement_base_path):
			handler = measurements.MeasurementHandler(measurement_name)
			try:
				collected_charge_beta_scan_single_voltage(
					handler.measurement_base_path,
					force = False,
					n_bootstrap = 11,
				)
				df = pandas.read_csv(handler.measurement_base_path/Path('collected_charge_beta_scan_single_voltage/results.csv'))
			except FileNotFoundError as e:
				warnings.warn(f'Cannot read data from measurement {repr(measurement_name)}, reason: {e}')
				continue
			for device_name in set(df['Device name']):
				collected_charge_data.append(
					{
						'Collected charge (V s) x_mpv value_on_data': float(df.query(f'`Device name`=="{device_name}"').query('Variable=="Collected charge (V s) x_mpv"').query('Type=="fit to original data"')['Value']),
						'Collected charge (V s) x_mpv mean': df.query(f'`Device name`=="{device_name}"').query('Variable=="Collected charge (V s) x_mpv"')['Value'].mean(),
						'Collected charge (V s) x_mpv std': df.query(f'`Device name`=="{device_name}"').query('Variable=="Collected charge (V s) x_mpv"')['Value'].std(),
						'Collected charge (V s) x_mpv median': df.query(f'`Device name`=="{device_name}"').query('Variable=="Collected charge (V s) x_mpv"')['Value'].median(),
						'Collected charge (V s) x_mpv MAD_std': median_abs_deviation(df.query(f'`Device name`=="{device_name}"').query('Variable=="Collected charge (V s) x_mpv"')['Value']),
						'Measurement name': measurement_name,
						'Bias voltage (V)': int(get_voltage_from_measurement(measurement_name)[:-1]),
						'Device name': device_name,
					}
				)
		collected_charge_df = pandas.DataFrame.from_records(collected_charge_data)
		collected_charge_df = collected_charge_df.sort_values(by='Bias voltage (V)')
		
		fig = plotly_utils.line(
			title = f'Collected charge vs bias voltage with beta source<br><sup>Measurement: {Vitorino.measurement_name}</sup>',
			data_frame = collected_charge_df,
			x = 'Bias voltage (V)',
			y = 'Collected charge (V s) x_mpv value_on_data',
			error_y = 'Collected charge (V s) x_mpv std',
			error_y_mode = 'band',
			hover_data = sorted(collected_charge_df),
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
		
		if Coulomb_calibration is not None:
			conversion_factor = read_calibration_factor(Coulomb_calibration/Path('create_calibration_for_Coulomb_conversion_in_beta_setup/conversion_factor (Coulomb over Volt over second).csv'))
			collected_charge_df['Collected charge (C)'] = collected_charge_df['Collected charge (V s) x_mpv median']*conversion_factor['mean (C/V/s)']
			collected_charge_df['Collected charge (C) std'] = ((collected_charge_df['Collected charge (V s) x_mpv MAD_std']*conversion_factor['mean (C/V/s)'])**2 + (collected_charge_df['Collected charge (V s) x_mpv median']*conversion_factor['std (C/V/s)'])**2)**.5
			fig = plotly_utils.line(
				title = f'Collected charge vs bias voltage with beta source<br><sup>Measurement: {Vitorino.measurement_name}</sup>',
				data_frame = collected_charge_df,
				x = 'Bias voltage (V)',
				y = 'Collected charge (C)',
				error_y = 'Collected charge (C) std',
				error_y_mode = 'band',
				hover_data = sorted(collected_charge_df),
				markers = 'circle',
				color = 'Device name',
			)
			fig.write_html(
				str(Vitorino.processed_data_dir_path/Path('collected charge vs bias voltage Coulomb.html')),
				include_plotlyjs = 'cdn',
			)

		collected_charge_df.to_csv(Vitorino.processed_data_dir_path/Path('collected_charge_vs_bias_voltage.csv'), index=False)

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
	parser.add_argument(
		'--Coulomb-calibration',
		metavar = 'path',
		help = 'Path to the base directory of a Coulomb calibration "measurement" produced by the script `create_calibration_for_Coulomb_conversion_in_beta_setup.py`.',
		required = False,
		dest = 'coulomb_calibration',
		type = str,
	)
	args = parser.parse_args()
	script_core(
		directory = Path(args.directory),
		Coulomb_calibration = Path(args.coulomb_calibration),
		force = True,
	)
