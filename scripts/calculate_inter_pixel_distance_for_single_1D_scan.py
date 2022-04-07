import pandas
import utils
from bureaucrat.Bureaucrat import Bureaucrat # https://github.com/SengerM/bureaucrat
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import measurements
from scipy import interpolate
from grafica.plotly_utils.utils import line

def calculate_interpixel_distance_by_linear_interpolation_using_normalized_collected_charge(data_df, threshold_percent:float, rough_inter_pixel_distance:float):
	"""Calculates the inter-pixel distance at a given threshold.
	
	Parameters
	----------
	data_df: pandas.DataFrame
		Data produced in a TCT 1D scan.
	threshold_percent: float
		Percent of the normalized collected charge height at which to
		calculate the inter-pixel distance. Usually you want 50 %.
	rough_inter_pixel_distance: float
		In meters. No need to be too precise, this is just to define the
		window of the data to use.
	"""
	df = data_df.query('n_pulse==1') # Use only the first pulse.
	df = utils.mean_std(df, by=['Distance (m)','Pad','n_pulse','n_position'])
	threshold_distance_for_each_pad = {}
	for pad in {'left','right'}:
		if pad == 'left':
			rows = (df['Distance (m)'] > df['Distance (m)'].mean() - rough_inter_pixel_distance) & (df['Pad']==pad)
		elif pad == 'right':
			rows = (df['Distance (m)'] < df['Distance (m)'].mean() + rough_inter_pixel_distance) & (df['Pad']==pad)
		distance_vs_charge_linear_interpolation = interpolate.interp1d(
			x = df.loc[rows,'Normalized collected charge median'],
			y = df.loc[rows,'Distance (m)'],
		)
		threshold_distance_for_each_pad[pad] = distance_vs_charge_linear_interpolation(threshold_percent/100)
	return {
		'Inter-pixel distance (m)': threshold_distance_for_each_pad['right']-threshold_distance_for_each_pad['left'],
		'Threshold (%)': threshold_percent,
		'Left pad distance (m)': threshold_distance_for_each_pad['left'],
		'Right pad distance (m)': threshold_distance_for_each_pad['right'],
	}

def script_core(directory:Path, rough_inter_pixel_distance:float, window_size:float, number_of_bootstrap_replicas_to_estimate_uncertainty:int, force:bool=False):
	bureaucrat = Bureaucrat(
		directory,
		new_measurement = False,
		variables = locals(),
	)
	
	if force == False and bureaucrat.job_successfully_completed_by_script('this script'):
		return
	
	with bureaucrat.verify_no_errors_context():
		for required_script in {'parse_waveforms_from_scan.py'}:
			if not bureaucrat.job_successfully_completed_by_script(required_script):
				raise RuntimeError(f'There is no previous successfull run of `{required_script}` for measurement {bureaucrat.measurement_name}, cannot proceed.')
		
		measurement_handler = measurements.MeasurementHandler(bureaucrat.measurement_name)
		measurement_handler.tag_left_and_right_pads()
		if 'Normalized collected charge' not in measurement_handler.measurement_data.columns:
			normalized_charge_df = utils.calculate_normalized_collected_charge(
				measurement_handler.measurement_data,
				window_size = window_size,
			)
			measurement_handler.measurement_data['Normalized collected charge'] = normalized_charge_df['Normalized collected charge']
		
		data_df = measurement_handler.measurement_data # Shorthand
		data_df = data_df.query('n_pulse == 1') # Use only data from pulse 1 so let's discard all the rest from now on...
		
		interpixel_distances_df = pandas.DataFrame()
		for threshold in sorted({8,22,37,50,60+3,77,92}):
			calculated_values = calculate_interpixel_distance_by_linear_interpolation_using_normalized_collected_charge(data_df, threshold_percent=threshold, rough_inter_pixel_distance=rough_inter_pixel_distance)
			interpixel_distances_df = interpixel_distances_df.append(calculated_values, ignore_index=True)
			if threshold == 50: # Bootstrap IPD ---
				bootstrapped_IPDs = [None]*number_of_bootstrap_replicas_to_estimate_uncertainty
				for k in range(len(bootstrapped_IPDs)):
					fake_IPD = calculate_interpixel_distance_by_linear_interpolation_using_normalized_collected_charge(
						utils.resample_measured_data(data_df), 
						threshold_percent = threshold,
						rough_inter_pixel_distance=rough_inter_pixel_distance,
					)['Inter-pixel distance (m)']
					bootstrapped_IPDs[k] = fake_IPD
		interpixel_distances_df.set_index('Threshold (%)', inplace=True)
		
		with open(bureaucrat.processed_data_dir_path/Path('interpixel_distance.txt'), 'w') as ofile:
			print(f'Inter-pixel distance (m) = {calculate_interpixel_distance_by_linear_interpolation_using_normalized_collected_charge(data_df, threshold_percent=50, rough_inter_pixel_distance=rough_inter_pixel_distance)["Inter-pixel distance (m)"]}', file=ofile)
		
		fig = line(
			data_frame = utils.mean_std(data_df, by=['Distance (m)','Pad']),
			x = 'Distance (m)',
			y = 'Normalized collected charge median',
			error_y = 'Normalized collected charge MAD_std',
			error_y_mode = 'band',
			color = 'Pad',
			labels = {
				'Normalized collected charge median': 'Normalized collected charge',
			},
			title = f'Inter-pixel distance<br><sup>Measurement: {bureaucrat.measurement_name}</sup>',
		)
		annotations = []
		for threshold in interpixel_distances_df.index:
			arrow = go.layout.Annotation(
				dict(
					x = interpixel_distances_df.loc[threshold, 'Right pad distance (m)'],
					y = threshold/100,
					ax = interpixel_distances_df.loc[threshold, 'Left pad distance (m)'],
					ay = threshold/100,
					xref = "x", 
					yref = "y",
					showarrow = True,
					axref = "x", ayref='y',
					arrowhead = 3,
					arrowwidth = 1.5,
					arrowcolor = 'black' if int(threshold)==50 else 'gray',
				)
			)
			annotations.append(arrow)
			text = go.layout.Annotation(
				dict(
					ax = (interpixel_distances_df.loc[threshold, 'Left pad distance (m)']+interpixel_distances_df.loc[threshold, 'Right pad distance (m)'])/2,
					ay = threshold/100,
					x = (interpixel_distances_df.loc[threshold, 'Left pad distance (m)']+interpixel_distances_df.loc[threshold, 'Right pad distance (m)'])/2,
					y = threshold/100,
					xref = "x", 
					yref = "y",
					text = f'{interpixel_distances_df.loc[threshold,"Inter-pixel distance (m)"]*1e6:.1f} µm<br> ',
					axref = "x", ayref='y',
					font = {'color': 'black' if int(threshold)==50 else 'gray'},
				)
			)
			annotations.append(text)
		fig.update_layout(annotations = annotations)
		fig.write_html(str(bureaucrat.processed_data_dir_path/Path('inter_pixel_distance.html')), include_plotlyjs='cdn')
		
		with open(bureaucrat.processed_data_dir_path/Path('interpixel_distance_bootstrapped_values.txt'), 'w') as ofile:
			for ipd in bootstrapped_IPDs:
				print(f'{ipd}', file=ofile)
		df = pandas.DataFrame({'IPD (m)': bootstrapped_IPDs})
		fig = px.histogram(
			title = f'Bootstrapped replicas of the inter-pixel distance<br><sup>{bureaucrat.measurement_name}</sup>',
			data_frame = df, 
			x = 'IPD (m)', 
			marginal = 'rug', 
			nbins = int((df['IPD (m)'].max()-df['IPD (m)'].min())/.1e-6)
		)
		fig.write_html(str(bureaucrat.processed_data_dir_path/Path('bootstrapped_IPD_distribution.html')), include_plotlyjs='cdn')

if __name__ == '__main__':
	import argparse
	
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--dir',
		metavar = 'path', 
		help = 'Path to the base directory of a measurement.',
		required = True,
		dest = 'directory',
		type = str,
	)
	args = parser.parse_args()
	script_core(
		Path(args.directory), 
		rough_inter_pixel_distance = 100e-6,
		window_size = 300e-6, # From the microscope pictures.
		number_of_bootstrap_replicas_to_estimate_uncertainty = 33,
		force = True
	)
