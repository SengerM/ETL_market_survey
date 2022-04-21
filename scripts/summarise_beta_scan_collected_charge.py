import pandas
from bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import warnings
import grafica.plotly_utils.utils as plotly_utils
import numpy

from utils import read_measurement_list, get_voltage_from_measurement

def script_core(directory: Path, dut_name: str, force: bool=False, average_voltage=None, charge_measurement=None, active_thickness=50):
	Vitorino = Bureaucrat(
		directory,
		new_measurement = False,
		variables = locals(),
	)

	if force == False and Vitorino.job_successfully_completed_by_script('this script'):
		return

	conversion_factor = None
	if charge_measurement is not None:
		pcklpath = None
		if (charge_measurement/Path("average_collected_charge/.script_successfully_applied")).is_file():
			pcklpath = charge_measurement/Path("average_collected_charge/average_collected_charge.pckl")
		elif (charge_measurement/Path("summarise_beta_scan_collected_charge/.script_successfully_applied")).is_file():
			pcklpath = charge_measurement/Path("summarise_beta_scan_collected_charge/average_collected_charge.pckl")
		if pcklpath is not None:
			with open(pcklpath, 'rb') as pcklfile:
				import math
				import pickle
				from scipy.constants import elementary_charge
				mean, stddev = pickle.load(pcklfile)
				PIN_charge = elementary_charge * ((31*math.log(active_thickness) + 128) * active_thickness)/3.65
				conversion_factor = (PIN_charge/mean, stddev * PIN_charge/(mean**2))


	with Vitorino.verify_no_errors_context():
		collected_charge_data = []
		charge_data = []
		for measurement_name in read_measurement_list(Vitorino.measurement_base_path):
			if not (Vitorino.measurement_base_path.parent/Path(measurement_name)/Path('plot_collected_charge/.script_successfully_applied')).is_file():
				warnings.warn(f"Collected Charge plotter was not successfully completed for {measurement_name}")
				continue
			try:
				df = pandas.read_csv(Vitorino.measurement_base_path.parent/Path(measurement_name)/Path('plot_collected_charge/results.csv'))
			except FileNotFoundError:
				warnings.warn(f'Cannot read data from measurement {repr(measurement_name)}')
				continue
			collected_charge_data.append(
				{
					'Collected charge (V s) x_mpv': float(df.query(f'`Device name`=="{dut_name}"').query('Variable=="Collected charge (V s) x_mpv"').query('Type=="fit to data"')['Value']),
					'Measurement name': measurement_name,
					'Bias voltage (V)': int(get_voltage_from_measurement(measurement_name)[:-1]),
					'Fluence (neq/cm^2)/1e14': 0,
				}
			)
			if average_voltage is not None:
				if int(get_voltage_from_measurement(measurement_name)[:-1]) >= average_voltage:
					charge_data += [float(df.query(f'`Device name`=="{dut_name}"').query('Variable=="Collected charge (V s) x_mpv"').query('Type=="fit to data"')['Value'])]
		collected_charge_df = pandas.DataFrame.from_records(collected_charge_data)

		if conversion_factor is not None:
			collected_charge_df["Produced charge (C) x_mpv"] = conversion_factor[0] * collected_charge_df["Collected charge (V s) x_mpv"]

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

		if conversion_factor is not None:
			fig = plotly_utils.line(
				title = f'Produced charge vs bias voltage with beta source<br><sup>Measurement: {Vitorino.measurement_name}</sup>',
				data_frame = df,
				x = 'Bias voltage (V)',
				y = 'Produced charge (C) x_mpv',
				hover_data = sorted(df),
				markers = 'circle',
			)
			fig.write_html(
				str(Vitorino.processed_data_dir_path/Path('produced charge vs bias voltage.html')),
				include_plotlyjs = 'cdn',
			)

		collected_charge_df.to_csv(Vitorino.processed_data_dir_path/Path('collected_charge_vs_bias_voltage.csv'))

		if average_voltage is not None:
			charge_np = numpy.array(charge_data)
			charge_mean = numpy.mean(charge_np, axis=0)
			charge_stddev = numpy.std(charge_np, axis=0, ddof=1)

			print("The mean charge is {} +- {}".format(charge_mean, charge_stddev))

			#  Save a csv - for floats it is not perfect since there can be rounding, but this would only affect the least significant
			# digit and is probably not relevant, but for saving data without loss of precision, use the pickle instead
			with open(Vitorino.processed_data_dir_path/Path('average_collected_charge.csv'), 'w', newline='') as csvfile:
				import csv
				writer = csv.writer(csvfile, delimiter=',')
				writer.writerow(["mean",   charge_mean])
				writer.writerow(["stddev", charge_stddev])

			with open(Vitorino.processed_data_dir_path/Path('average_collected_charge.pckl'), 'wb') as pcklfile:
				import pickle
				pickle.dump([charge_mean, charge_stddev], pcklfile, protocol=-1)


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
	parser.add_argument('-a', '--average',
		metavar = "voltage",
		help = "The voltage above which the points should be considered for calculating the average collected charge",
		type = int,
		default = None,
	)
	parser.add_argument('-c', '--charge',
		metavar = 'path',
		dest = "charge",
		help = "Path to the directory containing the measurement of the average collected charge of the equivalent PIN device",
		type = str,
		default = None,
	)
	parser.add_argument('-t', '--thickness',
		metavar = "thickness",
		help = "The thickness (in um) to consider for the conversion to the charge produced by the device (Default: 50)",
		type = int,
		default = 50,
	)
	args = parser.parse_args()
	dut_name = str(input('Name of DUT? '))

	charge_measurement = None
	if args.charge is not None:
		charge_measurement = Path(args.charge)
	script_core(Path(args.directory), dut_name=dut_name, force=True, average_voltage=args.average, charge_measurement=charge_measurement, active_thickness=args.thickness)
