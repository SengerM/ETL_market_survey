import pandas
from bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def script_core(directory: Path, dut_name: str, force: bool=False):
	Nísia = Bureaucrat(
		directory,
		new_measurement = False,
		variables = locals(),
	)

	with Nísia.verify_no_errors_context():
		if Nísia.job_successfully_completed_by_script('beta_scan.py'):
			try:
				measured_data_df = pandas.read_feather(Nísia.processed_by_script_dir_path('beta_scan.py')/Path('measured_data.fd'))
			except FileNotFoundError:
				measured_data_df = pandas.read_csv(Nísia.processed_by_script_dir_path('beta_scan.py')/Path('measured_data.csv'))
			measured_data_df = measured_data_df.loc[measured_data_df['device_name']==dut_name]

			fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
			fig.update_layout(title =  f'Current/Voltage vs time with beta source<br><sup>Measurement: {Nísia.measurement_name}</sup>')
			for row_minus_one, variable in enumerate(['Bias current (A)', 'Bias voltage (V)']):
				fig.add_trace(
					go.Scatter(
						x = measured_data_df['When'],
						y = measured_data_df[variable],
						name = variable,
						mode = 'lines+markers',
					),
					row = row_minus_one + 1,
					col = 1
				)
				fig.update_yaxes(title_text=variable, row=row_minus_one+1, col=1)
			fig.update_xaxes(title_text='When', row=row_minus_one+1, col=1)
			fig.write_html(str(Nísia.processed_data_dir_path/Path(f'Power Supply.html')), include_plotlyjs='cdn')

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--dir',
		metavar = 'path',
		help = 'Path to the base directory of a beta scan',
		required = True,
		dest = 'directory',
		type = str,
	)
	args = parser.parse_args()
	dut_name = str(input('Name of DUT? '))
	script_core(Path(args.directory), dut_name=dut_name, force=True)