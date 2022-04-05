import pandas
import xml.etree.ElementTree as ET
from pathlib import Path
import dateutil
from bureaucrat.Bureaucrat import Bureaucrat
import datetime
from distutils.dir_util import copy_tree
from plotly.subplots import make_subplots
import plotly.graph_objects as go

MEASUREMENTS_DATA_DIR_PATH = Path.home()/Path('cernbox/projects/ETL_market_survey/measurements_data')

class ClariusRunBureaucrat:
	"""Use this to easily get the measured data and measurement name
	of a measurement made with Clarius in the probe station.
	"""
	def __init__(self, path):
		self.path = path
		xml_run = ET.parse(Path(path)/Path('run.xml')).getroot()
		rating = None
		relative_path_to_data = None
		for child in xml_run:
			if child.tag == 'Rating':
				rating = child.text
			if child.tag == 'DataFileRelativePath':
				relative_path_to_data = child.text
			if child.tag == 'Time':
				when = dateutil.parser.parse(child.text)
		self.data = {
			'id': xml_run.get('runId'),
			'name': xml_run.get('username') if xml_run.get('username') != '' else None,
			'rating': rating,
			'path to data': Path(path).parent/Path(relative_path_to_data.replace('\\','/')),
			'when': when,
		}
		
	def __getitem__(self, key):
		return self.data.get(key)
	
	@property
	def measured_data(self):
		if not hasattr(self, '_measured_data'):
			self._measured_data = pandas.read_excel(self['path to data'])
		return self._measured_data
	
	@property
	def name(self):
		return self.data['name']

def script_core(directory: Path):
	Claudio = ClariusRunBureaucrat(directory)
	
	John = Bureaucrat(
		(MEASUREMENTS_DATA_DIR_PATH/Path(f'{Claudio.name}_probe_station')).resolve(),
		variables = locals(),
		new_measurement = True,
	)
	
	with John.verify_no_errors_context():
		Claudio.measured_data['When'] = [Claudio.data['when'] + datetime.timedelta(0,seconds) for seconds in Claudio.measured_data['Time']]
		Claudio.measured_data.to_feather(John.processed_data_dir_path/Path(f'measured_data.fd'))
		
		copy_tree(str(directory.resolve()), str((John.processed_data_dir_path/Path('original_data_from_probestation')).resolve()))
		
		# Variables vs When plot ---
		CURRENT_ROW = 1
		VOLTAGE_ROW = 2
		df = Claudio.measured_data
		fig = make_subplots(rows=2, cols=1, shared_xaxes=True,) # One for currents, another one for voltages.
		fig.update_layout(title=f'Probe station measurement<br><sup>Measurement: {John.measurement_name}</sup>')
		fig.update_yaxes(title_text="Current (unknown units)", row=CURRENT_ROW, col=1)
		fig.update_yaxes(title_text="Voltage (unknown units)", row=VOLTAGE_ROW, col=1)
		fig.update_xaxes(title_text='When', row=max(CURRENT_ROW,VOLTAGE_ROW), col=1)
		for col in df.columns:
			if col.lower() in {'time','when'}:
				continue
			if any([s in col.lower() for s in {'current','(A)'}]):
				plot_in_row = CURRENT_ROW
			elif any([s in col.lower() for s in {'volt','(V)','voltage'}]):
				plot_in_row = VOLTAGE_ROW
			else:
				continue
			fig.add_trace(
				go.Scatter(
					x = df['When'],
					y = df[col],
					name = col,
				),
				col = 1,
				row = plot_in_row,
			)
		fig.write_html(
			str(John.processed_data_dir_path/Path('plot_vs_when.html')),
			include_plotlyjs = 'cdn',
		)
		
		# IV curves ---
		USE_AS_X_AXIS = 'back_side_voltage'
		fig = go.Figure()
		fig.update_layout(
			title = f'IV curves (?) in the probe station<br><sup>Measurement: {John.measurement_name}</sup>',
			xaxis_title = 'Bias voltage (V)',
			yaxis_title = 'Current (A)',
		)
		for col in df.columns:
			if col.lower() in {'time','when'} or any([s in col.lower() for s in {'volt','(V)','voltage'}]):
				continue
			fig.add_trace(
				go.Scatter(
					x = df[USE_AS_X_AXIS].abs(),
					y = df[col].abs(),
					name = col,
				)
			)
		fig.write_html(
			str(John.processed_data_dir_path/Path('maybe_iv_curves_lin.html')),
			include_plotlyjs = 'cdn',
		)
		fig.update_yaxes(type = 'log')
		fig.write_html(
			str(John.processed_data_dir_path/Path('maybe_iv_curves_log.html')),
			include_plotlyjs = 'cdn',
		)

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('--dir',
		metavar = 'path',
		help = 'Path to the base measurement directory made with the probe station.',
		required = True,
		dest = 'directory',
		type = str,
	)
	
	args = parser.parse_args()
	script_core(Path(args.directory))
