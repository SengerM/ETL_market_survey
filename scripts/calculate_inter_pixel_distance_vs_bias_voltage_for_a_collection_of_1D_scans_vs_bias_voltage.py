import pandas
from bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import warnings
from utils import read_measurement_list
import measurements
from grafica.plotly_utils.utils import line
import numpy as np
import utils

from fit_erf_and_calculate_calibration_factor import script_core as fit_erf_and_calculate_calibration_factor
from calculate_inter_pixel_distance_for_single_1D_scan import script_core as calculate_inter_pixel_distance_for_single_1D_scan

def get_color(colorscale_name, loc):
	# https://stackoverflow.com/a/69702468/8849755
	from _plotly_utils.basevalidators import ColorscaleValidator
	
	def get_continuous_color(colorscale, intermed):
		"""
		Plotly continuous colorscales assign colors to the range [0, 1]. This function computes the intermediate
		color for any value in that range.

		Plotly doesn't make the colorscales directly accessible in a common format.
		Some are ready to use:
		
			colorscale = plotly.colors.PLOTLY_SCALES["Greens"]

		Others are just swatches that need to be constructed into a colorscale:

			viridis_colors, scale = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Viridis)
			colorscale = plotly.colors.make_colorscale(viridis_colors, scale=scale)

		:param colorscale: A plotly continuous colorscale defined with RGB string colors.
		:param intermed: value in the range [0, 1]
		:return: color in rgb string format
		:rtype: str
		"""
		import plotly.colors
		from PIL import ImageColor

		if len(colorscale) < 1:
			raise ValueError("colorscale must have at least one color")

		hex_to_rgb = lambda c: "rgb" + str(ImageColor.getcolor(c, "RGB"))

		if intermed <= 0 or len(colorscale) == 1:
			c = colorscale[0][1]
			return c if c[0] != "#" else hex_to_rgb(c)
		if intermed >= 1:
			c = colorscale[-1][1]
			return c if c[0] != "#" else hex_to_rgb(c)

		for cutoff, color in colorscale:
			if intermed > cutoff:
				low_cutoff, low_color = cutoff, color
			else:
				high_cutoff, high_color = cutoff, color
				break

		if (low_color[0] == "#") or (high_color[0] == "#"):
			# some color scale names (such as cividis) returns:
			# [[loc1, "hex1"], [loc2, "hex2"], ...]
			low_color = hex_to_rgb(low_color)
			high_color = hex_to_rgb(high_color)

		return plotly.colors.find_intermediate_color(
			lowcolor=low_color,
			highcolor=high_color,
			intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
			colortype="rgb",
		)
	cv = ColorscaleValidator("colorscale", "")
	# colorscale will be a list of lists: [[loc1, "rgb1"], [loc2, "rgb2"], ...] 
	colorscale = cv.validate_coerce(colorscale_name)
	
	if hasattr(loc, "__iter__"):
		return [get_continuous_color(colorscale, x) for x in loc]
	return get_continuous_color(colorscale, loc)

def script_core(directory: Path, window_size:float, force: bool=False):
	Adérito = Bureaucrat(
		directory,
		new_measurement = False,
		variables = locals(),
	)
	
	if force == False and Adérito.job_successfully_completed_by_script('this script'):
		return

	with Adérito.verify_no_errors_context():
		data_df = pandas.DataFrame(columns = ['Bias voltage (V) median','Bias voltage (V) MAD_std','Inter-pixel distance (m)','Distance calibration factor'])
		scans_1D = []
		for measurement_name in read_measurement_list(Adérito.measurement_base_path):
			fit_erf_and_calculate_calibration_factor(
				Adérito.measurement_base_path.parent/Path(measurement_name),
				force = False,
				window_size = 300e-6, # From the microscope pictures.
			)
			calculate_inter_pixel_distance_for_single_1D_scan(
				Adérito.measurement_base_path.parent/Path(measurement_name),
				force = False,
				window_size = 300e-6, # From the microscope pictures.
				rough_inter_pixel_distance = 100e-6,
				number_of_bootstrap_replicas_to_estimate_uncertainty = 33,
			)
			# Here we are sure that things were calculated, so now we can put everything together ---
			measurement_handler = measurements.MeasurementHandler(measurement_name)
			measurement_handler.measurement_data['measurement_name'] = measurement_name
			measurement_handler.measurement_data['Distance from center (m)'] = measurement_handler.measurement_data['Distance (m)'] - measurement_handler.distance_calibration['offset before scale']
			
			this_measurement_stuff = {
				'Bias voltage (V) median': np.abs(measurement_handler.bias_voltage_summary['median']),
				'Bias voltage (V) MAD_std': measurement_handler.bias_voltage_summary['MAD_std'],
				'Inter-pixel distance (m)': measurement_handler.inter_pixel_distance_summary['Inter-pixel distance (m) value on data'],
				'Inter-pixel distance (m) MAD_std': measurement_handler.inter_pixel_distance_summary['Inter-pixel distance (m) MAD_std'],
				'Distance scale factor': measurement_handler.distance_calibration['scale factor'],
				'measurement_name': measurement_name,
			}
			data_df = pandas.concat([data_df, pandas.DataFrame(this_measurement_stuff, index=[0])], ignore_index = True)
			# Get the full data for each scan so I can then plot them all together ---
			measurement_handler.tag_left_and_right_pads()
			measurement_handler.measurement_data['Normalized collected charge'] = utils.calculate_normalized_collected_charge(
				measurement_handler.measurement_data,
				window_size = window_size,
			)
			scans_1D.append(measurement_handler.measurement_data.query('n_pulse==1'))
		scans_1D = pandas.concat(scans_1D, ignore_index=True)
		
		data_df = data_df.sort_values(by='Bias voltage (V) median')
		for col in {'Inter-pixel distance (m)','Inter-pixel distance (m) MAD_std'}:
			data_df[f'{col} calibrated'] = data_df[col]*data_df['Distance scale factor']
		
		data_df.to_csv(Adérito.processed_data_dir_path/Path('inter_pixel_distance_vs_bias_voltage.csv'))
		
		for y_var in {'',' calibrated'}:
			fig = line(
				title = f'Inter pixel distance<br><sup>Measurement: {Adérito.measurement_name}</sup>',
				data_frame = data_df,
				x = 'Bias voltage (V) median',
				y = f'Inter-pixel distance (m){y_var}',
				error_y = f'Inter-pixel distance (m) MAD_std{y_var}',
				error_y_mode = 'band',
				markers = True,
				labels = {
					'Inter-pixel distance (m)': 'Inter-pixel distance (m) without calibration',
					'Inter-pixel distance (m) calibrated': 'Inter-pixel distance (m)',
					'Bias voltage (V) median': 'Bias voltage (V)',
				},
			)
			fig.write_html(
				str(Adérito.processed_data_dir_path/Path(f'inter pixel distance{y_var}.html')),
				include_plotlyjs = 'cdn',
			)
		
		fig = line(
			title = f'Distance scale calibration factor<br><sup>Measurement: {Adérito.measurement_name}</sup>',
			data_frame = data_df,
			x = 'Bias voltage (V) median',
			y = f'Distance scale factor',
			markers = True,
			labels = {
				'Inter-pixel distance (m)': 'Inter-pixel distance (m) without calibration',
				'Inter-pixel distance (m) calibrated': 'Inter-pixel distance (m)',
				'Bias voltage (V) median': 'Bias voltage (V)',
			},
		)
		fig.write_html(
			str(Adérito.processed_data_dir_path/Path(f'distance calibration factor.html')),
			include_plotlyjs = 'cdn',
		)
		
		DIRECTORY_FOR_SCAN_PLOTS = Adérito.processed_data_dir_path/Path('scans vs voltage plots')
		DIRECTORY_FOR_SCAN_PLOTS.mkdir(exist_ok=True)
		scans_1D = utils.mean_std(scans_1D, by=['measurement_name','n_position','Pad','n_pulse'])
		scans_1D = scans_1D.set_index('measurement_name')
		data_df = data_df.set_index('measurement_name')
		scans_1D['Bias voltage (V)'] = data_df['Bias voltage (V) median'].astype(int)
		for var in {'Distance from center (m)','Distance (m)'}:
			for stat in {'mean','std','median','MAD_std'}:
				col = f'{var} {stat}'
				scans_1D[f'{col} calibrated'] = scans_1D[col]*data_df['Distance scale factor']
		scans_1D = scans_1D.reset_index()
		scans_1D = scans_1D.sort_values(by=['Bias voltage (V)','Pad','Distance (m) mean'])
		for variable in {'Collected charge (V s)','Normalized collected charge','t_50 (s)','Amplitude (V)','Noise (V)','Rise time (s)','Time over noise (s)'}:
			fig = line(
				title = f'{variable} vs position for different voltages<br><sup>Measurement: {Adérito.measurement_name}</sup>',
				data_frame = scans_1D,
				x = 'Distance from center (m) mean calibrated',
				y = f'{variable} median',
				error_y = f'{variable} MAD_std',
				error_y_mode = 'band',
				color = 'Bias voltage (V)',
				line_dash = 'Pad',
				line_group = 'measurement_name',
				color_discrete_sequence = [get_color('Plasma',i/len(set(scans_1D['measurement_name']))) for i in range(len(set(scans_1D['measurement_name'])))],
				labels = {
					f'{variable} median': variable,
					'Distance (m) mean': 'Distance (m)',
					'Distance from center (m) mean calibrated': 'Distance from center after calibration (m)',
				}
			)
			for x in [-window_size/2, window_size/2]:
				fig.add_vline(x = x, line_dash='dash')
			fig.write_html(
				str(DIRECTORY_FOR_SCAN_PLOTS/Path(f'{variable} vs distance vs bias voltage.html')),
				include_plotlyjs = 'cdn',
			)
		
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
	script_core(
		Path(args.directory), 
		window_size = 300e-6, # From the microscope pics.
		force = True
	)
