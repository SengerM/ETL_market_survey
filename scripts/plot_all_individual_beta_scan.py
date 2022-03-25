import pandas
from bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import plotly.express as px
import numpy as np
import warnings
import grafica.plotly_utils.utils as plotly_utils

from utils import read_measurement_list

from plot_power_supply import script_core as plot_power_supply
from plot_everything_beta_point import script_core as plot_everything_beta_point
from clean_beta_scan import script_core as clean_beta_scan
from plot_time_resolution import script_core as plot_time_resolution

def script_core(directory: Path, dut_name: str, force: bool=False):
	Adérito = Bureaucrat(
		directory,
		new_measurement = False,
		variables = locals(),
	)

	if force == False and Adérito.job_successfully_completed_by_script('this script'):
		return

	with Adérito.verify_no_errors_context():
		for measurement_name in read_measurement_list(Adérito.measurement_base_path):
			if not (Adérito.measurement_base_path.parent/Path(measurement_name)/Path('beta_scan/.script_successfully_applied')).is_file():
				warnings.warn(f'Beta "Scan" was not successfully completed for {measurement_name}')
				continue
			print("Running on {}".format(measurement_name))

			print("  Plotting the I/V over time")
			plot_power_supply(Adérito.measurement_base_path.parent/Path(measurement_name), dut_name)

			print("  Making all the individual plots for the beta timing point")
			plot_everything_beta_point(Adérito.measurement_base_path.parent/Path(measurement_name))

			if (Adérito.measurement_base_path.parent/Path(measurement_name)/Path('cuts.csv')).is_file():
				print("  Cleaning triggers that do not pass the quality criteria")
				clean_beta_scan(Adérito.measurement_base_path.parent/Path(measurement_name))

			print("  Creating the timing plots")
			plot_time_resolution(Adérito.measurement_base_path.parent/Path(measurement_name), force=force)


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
	dut_name = str(input('Name of DUT? '))
	script_core(Path(args.directory), dut_name=dut_name, force=True)
