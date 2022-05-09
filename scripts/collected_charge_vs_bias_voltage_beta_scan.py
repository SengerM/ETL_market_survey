import pandas
from bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import warnings
import grafica.plotly_utils.utils as plotly_utils
import numpy
from scipy.stats import median_abs_deviation
from utils import read_measurement_list, get_voltage_from_measurement
from collected_charge_beta_scan_single_voltage import script_core as collected_charge_beta_scan_single_voltage
import measurements

def script_core(directory:Path, force:bool=False, charge_calibration_measurement:Path=None, active_thickness:float=None, average_voltage=None):
	Vitorino = Bureaucrat(
		directory,
		new_measurement = False,
		variables = locals(),
	)
	
	if measurements.measurement_type(Vitorino.measurement_name) != 'beta voltage scan':
		raise NotImplementedError(f'Dont know how to process a measurement of type {repr(measurements.measurement_type(Vitorino.measurement_name))}.')

	if force == False and Vitorino.job_successfully_completed_by_script('this script'):
		return
	
	conversion_factor = None
	if charge_calibration_measurement is not None:
		if (charge_calibration_measurement/Path("average_collected_charge/.script_successfully_applied")).is_file():
			pcklpath = charge_calibration_measurement/Path("average_collected_charge/average_collected_charge.pckl")
		elif (charge_calibration_measurement/Path("summarise_beta_scan_collected_charge/.script_successfully_applied")).is_file():
			pcklpath = charge_calibration_measurement/Path("summarise_beta_scan_collected_charge/average_collected_charge.pckl")
		else:
			raise RuntimeError(f'Cannot find `charge_calibration_measurement` data in {charge_calibration_measurement}.')
		with open(pcklpath, 'rb') as pcklfile:
			import math
			import pickle
			from scipy.constants import elementary_charge
			devices, means, stddevs = pickle.load(pcklfile)
			PIN_charge_theory_Coulomb = elementary_charge * ((31*math.log(active_thickness*1e6) + 128) * active_thickness*1e6)/3.65
			conversion_factor = {}
			for i in range(len(devices)):
				conversion_factor[devices[i]] = (PIN_charge_theory_Coulomb/means[i], stddevs[i] * PIN_charge_theory_Coulomb/(means[i]**2))
	
	with Vitorino.verify_no_errors_context():
		collected_charge_data = []
		charge_data = {}
		for measurement_name in read_measurement_list(Vitorino.measurement_base_path):
			handler = measurements.MeasurementHandler(measurement_name)
			try:
				collected_charge_beta_scan_single_voltage(
					handler.measurement_base_path,
					force = True,
					n_bootstrap = 0,
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
						'Fluence (neq/cm^2)/1e14': 0,
						'Device name': device_name,
					}
				)
				if average_voltage is not None:
					if int(get_voltage_from_measurement(measurement_name)[:-1]) >= average_voltage:
						if device_name not in charge_data:
							charge_data[device_name] = []
						charge_data[device_name] += [float(df.query(f'`Device name`=="{device_name}"').query('Variable=="Collected charge (V s) x_mpv"').query('Type=="fit to data"')['Value'])]
		collected_charge_df = pandas.DataFrame.from_records(collected_charge_data)

		# Note: If a device has no conversion factor... the output (produced charge) will be set to 0
		if conversion_factor is not None:
			device_names = set(collected_charge_df['Device name'])
			conditions = []
			values_conversion = []
			values_stddev = []
			for device in device_names:
				conditions += [collected_charge_df['Device name'] == device]
				if device in conversion_factor:
					values_conversion += [conversion_factor[device][0]]
					values_stddev += [conversion_factor[device][1]]
				else:
					values_conversion += [0]
					values_stddev += [0]
			conversion = numpy.select(conditions, values_conversion)
			stddev     = numpy.select(conditions, values_stddev)

			collected_charge_df["Produced charge (C) x_mpv value_on_data"] = conversion * collected_charge_df["Collected charge (V s) x_mpv value_on_data"]
			collected_charge_df["Produced charge (C) x_mpv std"]           = ((conversion * collected_charge_df["Collected charge (V s) x_mpv std"])**2 + (stddev * collected_charge_df["Collected charge (V s) x_mpv value_on_data"])**2)**(1/2)

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

		if conversion_factor is not None:
			fig = plotly_utils.line(
				title = f'Produced charge vs bias voltage with beta source<br><sup>Measurement: {Vitorino.measurement_name}</sup>',
				data_frame = df,
				x = 'Bias voltage (V)',
				y = 'Produced charge (C) x_mpv value_on_data',
				error_y = 'Produced charge (C) x_mpv std',
				hover_data = sorted(df),
				markers = 'circle',
				labels = {
					'Produced charge (C) x_mpv value_on_data': 'Produced charge (C)',
				},
				color = 'Device name',
			)
			fig.write_html(
				str(Vitorino.processed_data_dir_path/Path('produced charge vs bias voltage.html')),
				include_plotlyjs = 'cdn',
			)

		collected_charge_df.to_csv(Vitorino.processed_data_dir_path/Path('collected_charge_vs_bias_voltage.csv'))

		if average_voltage is not None:
			devices = []
			charge_array = []
			for device in charge_data:
				devices += [device]
				charge_array += [charge_data[device]]
			charge_np = numpy.array(charge_array)
			charge_mean = numpy.mean(charge_np, axis=1)
			charge_stddev = numpy.std(charge_np, axis=1, ddof=1)

			for i in range(len(devices)):
				print("The mean charge for {} is {} +- {}".format(devices[i], charge_mean[i], charge_stddev[i]))

			#  Save a csv - for floats it is not perfect since there can be rounding, but this would only affect the least significant
			# digit and is probably not relevant, but for saving data without loss of precision, use the pickle instead
			with open(Vitorino.processed_data_dir_path/Path('average_collected_charge.csv'), 'w', newline='') as csvfile:
				import csv
				writer = csv.writer(csvfile, delimiter=',')
				for i in range(len(devices)):
					writer.writerow(["{} mean".format(devices[i]),   charge_mean[i]])
					writer.writerow(["{} stddev".format(devices[i]), charge_stddev[i]])

			with open(Vitorino.processed_data_dir_path/Path('average_collected_charge.pckl'), 'wb') as pcklfile:
				import pickle
				pickle.dump([devices, charge_mean, charge_stddev], pcklfile, protocol=-1)

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
		help = "The voltage above which the points should be considered for calculating the average collected charge.",
		type = int,
		default = None,
	)
	parser.add_argument('-c', '--charge-calibration-measurement',
		metavar = 'path',
		dest = "charge",
		help = "Path to the directory containing the measurement of the average collected charge of the equivalent PIN device.",
		type = str,
		default = None,
	)
	parser.add_argument('-t', '--thickness',
		metavar = "thickness",
		help = "The thickness (in meters) to consider for the conversion to the charge produced by the device (Default: 50e-6).",
		type = float,
		default = 50e-6,
	)
	args = parser.parse_args()
	charge_calibration_measurement = None
	if args.charge is not None:
		charge_calibration_measurement = Path(args.charge)
	script_core(
		Path(args.directory), 
		force = True, 
		average_voltage = args.average, 
		charge_calibration_measurement = charge_calibration_measurement, 
		active_thickness = args.thickness,
	)
