from bureaucrat.Bureaucrat import Bureaucrat # https://github.com/SengerM/bureaucrat
from pathlib import Path
import pandas
from measurements import MEASUREMENTS_DATA_PATH
import pickle
import math
import csv
import plotly.express as px

def script_core(measurements=None, plot=False):
	if measurements is None:
		print("You must specify a dictionary of measurements with labels")
		return
	if type(measurements) is not type({}):
		print("The measurements parameter must by a python dict")
		return

	KalEl = Bureaucrat(
		MEASUREMENTS_DATA_PATH/Path("Average_Charge"),
        new_measurement = True,
		variables = locals(),
	)

	with KalEl.verify_no_errors_context():
		df = pandas.DataFrame(columns = {'Mean charge (C)', 'Stddev charge (C)', 'label'})
		all_data = []
		count = 0
		all_mean = 0
		all_stddev = 0
		for key in measurements:
			if measurements[key] is None or not (MEASUREMENTS_DATA_PATH/Path(measurements[key]+"/summarise_beta_scan_collected_charge/.script_successfully_applied")).is_file():
				print("You must specify a valid measurement for {}".format(key))
				continue
			pcklpath = MEASUREMENTS_DATA_PATH/Path(measurements[key]+"/summarise_beta_scan_collected_charge/average_collected_charge.pckl")
			if not pcklpath.is_file():
				print("You must pass the option to calculate the average charge to the summarise_beta_scan_collected_charge.py script for \"{}\" in order to have a valid measurement".format(measurements[key]))
				continue
			with open(str(pcklpath), 'rb') as pcklfile:
				mean, stddev = pickle.load(pcklfile)
				count += 1
				all_mean += mean
				all_stddev += stddev**2
				all_data.append({
					'Measurement': key,
					'Mean charge (C)': mean,
					'Stddev charge (C)': stddev,
					'Label': "measurement",
				})
		if count != 0:
			all_mean /= count
			all_stddev = math.sqrt(all_stddev)/count

			with open(KalEl.processed_data_dir_path/Path('average_collected_charge.csv'), 'w', newline='') as csvfile:
				writer = csv.writer(csvfile, delimiter=',')
				writer.writerow(["mean",   all_mean])
				writer.writerow(["stddev", all_stddev])

			with open(KalEl.processed_data_dir_path/Path('average_collected_charge.pckl'), 'wb') as pcklfile:
				pickle.dump([all_mean, all_stddev], pcklfile, protocol=-1)


			all_data.append({
				'Measurement': "average",
				'Mean charge (C)': all_mean,
				'Stddev charge (C)': all_stddev,
				'Label': "average",
			})
			df = pandas.DataFrame.from_records(all_data)

			if plot:
				fig = px.bar(df, x="Measurement", y="Mean charge (C)", color="Label", error_y="Stddev charge (C)")
				fig.write_html(
					str(KalEl.processed_data_dir_path/Path('average collected charge.html')),
					include_plotlyjs = 'cdn',
				)

			df.to_feather(KalEl.processed_data_dir_path/Path('measured_data.fd'))
		else:
			print("There were no valid measurements, so the data is empty")

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='Calculates the average, between different measurements, of the average collected charge (i.e. the average collected charge option must have been passed and ran on each of the individual measurements)')
	parser.add_argument(
		'-p', '--plot',
		help = 'If this flag is passed, a histogram for comparing the individual measurements is plotted. Default is not plot',
		action = 'store_true',
		dest = 'plot',
	)

	args = parser.parse_args()

	# List the measurements to be averaged as pairs of LABEL: MEASUREMENT_DIRECTORY
	measurements = {
		"MS11PIN": "20220407154640_BetaScan_MS11_PIN_sweeping_bias_voltage",
		"MS12PIN": "20220412121103_BetaScan_MS12_PIN_sweeping_bias_voltage",
	}

	script_core(measurements, plot = args.plot)