import pandas
from bureaucrat.SmarterBureaucrat import SmarterBureaucrat # https://github.com/SengerM/bureaucrat
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import grafica.plotly_utils.utils as plotly_utils
from scipy.stats import median_abs_deviation
from scipy.optimize import curve_fit
import shutil

from utils import remove_nans_grouping_by_n_trigger, get_voltage_from_measurement

k_MAD_TO_STD = 1.4826 # https://en.wikipedia.org/wiki/Median_absolute_deviation#Relation_to_standard_deviation
N_BOOTSTRAP = 33

def gaussian(x, mu, sigma, amplitude=1):
	return amplitude/sigma/(2*np.pi)**.5*np.exp(-((x-mu)/sigma)**2/2)

def resample_measured_data(measured_data_df):
	resampled_df = measured_data_df.pivot(
		index = 'n_trigger',
		columns = 'device_name',
		values = list(set(measured_data_df.columns) - {'n_trigger','device_name'}),
	)
	resampled_df = resampled_df.sample(frac=1, replace=True)
	resampled_df = resampled_df.stack()
	resampled_df = resampled_df.reset_index()
	return resampled_df

def calculate_Delta_t_df(data_df):
	"""`data_df` must possess the following columns: 'n_trigger','n_pulse','t_10 (s)','t_20 (s)',...,'t_90 (s)'"""
	for col in {'n_trigger','n_pulse'}.union({f't_{i} (s)' for i in [10,20,30,40,50,60,70,80,90]}):
		if col not in data_df.columns:
			raise ValueError(f'{repr(col)} is not a column of `data_df`.')

	data_df = data_df.copy() # Don't want to touch the original...
	data_df.set_index('n_trigger', inplace=True)
	pulse_1_df = data_df.query(f'n_pulse==1')
	pulse_2_df = data_df.query(f'n_pulse==2')

	Delta_t_df = pandas.DataFrame()
	for k1 in [10,20,30,40,50,60,70,80,90]:
		for k2 in [10,20,30,40,50,60,70,80,90]:
			temp_df = pandas.DataFrame()
			temp_df['Δt (s)'] = pulse_1_df[f't_{k1} (s)'] - pulse_2_df[f't_{k2} (s)']
			temp_df['k_1 (%)'] = k1
			temp_df['k_2 (%)'] = k2
			temp_df.reset_index(inplace=True)
			temp_df.set_index(['n_trigger','k_1 (%)','k_2 (%)'], inplace=True)
			Delta_t_df = pandas.concat([Delta_t_df, temp_df])
	Delta_t_df.reset_index(inplace=True)
	Delta_t_df = Delta_t_df.dropna()
	return Delta_t_df

def calculate_Delta_t_fluctuations_df(Delta_t_df):
	Delta_t_fluctuations_df = Delta_t_df.groupby(by=['k_1 (%)','k_2 (%)']).agg(median_abs_deviation).reset_index()
	Delta_t_fluctuations_df.drop('n_trigger', axis=1, inplace=True)
	Delta_t_fluctuations_df.rename(columns={'Δt (s)': 'MAD(Δt) (s)'}, inplace=True)
	Delta_t_fluctuations_df['k MAD(Δt) (s)'] = Delta_t_fluctuations_df['MAD(Δt) (s)']*k_MAD_TO_STD
	return Delta_t_fluctuations_df.set_index(['k_1 (%)','k_2 (%)'])

def find_best_k1_k2(Delta_t_fluctuations_df):
	if Delta_t_fluctuations_df.index.names != ['k_1 (%)', 'k_2 (%)']:
		raise ValueError(f"The indices of `Delta_t_fluctuations_df` must be ['k_1 (%)', 'k_2 (%)'].")
	return Delta_t_fluctuations_df['k MAD(Δt) (s)'].idxmin() # k1, k2

def plot_cfd(Delta_t_fluctuations_df):
	pivot_table_df = pandas.pivot_table(
		Delta_t_fluctuations_df,
		values = 'k MAD(Δt) (s)',
		index = 'k_1 (%)',
		columns = 'k_2 (%)',
		aggfunc = np.mean,
	)
	fig = go.Figure(
		data = go.Contour(
			z = pivot_table_df.to_numpy(),
			x = pivot_table_df.index,
			y = pivot_table_df.columns,
			contours = dict(
				coloring ='heatmap',
				showlabels = True, # show labels on contours
			),
			colorbar = dict(
				title = 'k MAD(Δt)',
				titleside = 'right',
			),
			hovertemplate = 'k<sub>1</sub>: %{y:.0f} %<br>k<sub>2</sub>: %{x:.0f} %<br>k MAD(Δt): %{z:.2e} s',
			name = '',
		),
	)
	k1_min, k2_min = find_best_k1_k2(Delta_t_fluctuations_df)
	fig.add_trace(
		go.Scatter(
			x = [k2_min],
			y = [k1_min],
			mode = 'markers',
			hovertext = [f'<b>Minimum</b><br>k<sub>1</sub>: {k1_min:.0f} %<br>k<sub>2</sub>: {k2_min:.0f} %<br>k MAD(Δt): {Delta_t_fluctuations_df.loc[(k1_min,k2_min),"k MAD(Δt) (s)"]*1e12:.2f} ps'],
			hoverinfo = 'text',
			marker = dict(
				color = '#61ff5c',
			),
			name = '',
		)
	)
	fig.update_yaxes(
		scaleanchor = "x",
		scaleratio = 1,
	)
	fig.update_layout(
		xaxis_title = 'k<sub>2</sub> (%)',
		yaxis_title = 'k<sub>1</sub> (%)',
	)
	return fig

def draw_median_and_MAD_vertical_lines(plotlyfig, center: float, amplitude: float, text: str):
	plotlyfig.add_vline(
		x = center,
		line_color = 'black',
	)
	for s in [-amplitude, amplitude]:
		plotlyfig.add_vline(
			x = center + s,
			line_color = 'black',
			line_dash = 'dash',
		)
	plotlyfig.add_annotation(
		x = center,
		y = .25,
		ax = center + amplitude,
		ay = .25,
		yref = 'y domain',
		showarrow = True,
		axref = "x", ayref='y',
		arrowhead = 3,
		arrowwidth = 1.5,
	)
	plotlyfig.add_annotation(
		x = center + amplitude,
		y = .25,
		ax = center,
		ay = .25,
		yref = 'y domain',
		showarrow = True,
		axref = "x", ayref='y',
		arrowhead = 3,
		arrowwidth = 1.5,
	)
	plotlyfig.add_annotation(
		text = text,
		ax = center + amplitude/2,
		ay = .27,
		x = center + amplitude/2,
		y = .27,
		yref = "y domain",
		axref = "x", ayref='y',
	)

def plot_Delta_t_histogram(Delta_t_df, k1, k2):
	fig = grafica.new(
		xlabel = 'Δt (s)',
		ylabel = 'Number of events',
	)
	data_to_plot = Delta_t_df.loc[(Delta_t_df['k_1 (%)']==k1)&(Delta_t_df['k_2 (%)']==k2), 'Delta_t (s)']
	fig.histogram(
		samples = data_to_plot,
	)
	draw_median_and_MAD_vertical_lines(
		plotlyfig = fig.plotly_figure,
		median = np.median(data_to_plot),
		MAD = median_abs_deviation(data_to_plot)
	)
	return fig

def fit_gaussian_to_samples(samples, bins='auto'):
	hist, bins_edges = np.histogram(
		samples,
		bins = bins,
	)
	x_values = bins_edges[:-1] + np.diff(bins_edges)[0]/2
	y_values = hist
	try:
		popt, pcov = curve_fit(
			gaussian,
			x_values,
			y_values,
			p0 = [np.median(samples),median_abs_deviation(samples)*k_MAD_TO_STD,max(y_values)],
		)
		return popt[0], popt[1], popt[2]
	except RuntimeError: # This happens when the fit fails because there are very few samples.
		return float('NaN'),float('NaN'),float('NaN')

def process_single_voltage_measurement(directory:Path, force:bool=False):
	Norberto = SmarterBureaucrat(directory,	_locals = locals())
	
	Norberto.check_required_scripts_were_run_before('beta_scan.py')
	
	if force == False and Norberto.script_was_applied_without_errors(): # If this was already done, don't do it again...
		return
	
	with Norberto.do_your_magic():
		try:
			measured_data_df = pandas.read_feather(Norberto.path_to_output_directory_of_script_named('beta_scan.py')/Path('measured_data.fd'))
		except FileNotFoundError:
			measured_data_df = pandas.read_csv(Norberto.path_to_output_directory_of_script_named('beta_scan.py')/Path('measured_data.csv'))

		if Norberto.check_required_scripts_were_run_before('clean_beta_scan.py', raise_error=False): # If there was a cleaning done, let's take it into account...
			shutil.copyfile( # Put a copy of the cuts in the output directory so there is a record of what was done.
				Norberto.path_to_output_directory_of_script_named('clean_beta_scan.py')/Path('cuts.csv'),
				Norberto.path_to_default_output_directory/Path('cuts.csv')
			)
			df = pandas.read_feather(Norberto.path_to_output_directory_of_script_named('clean_beta_scan.py')/Path('clean_triggers.fd'))
			df = df.set_index('n_trigger')
			measured_data_df = remove_nans_grouping_by_n_trigger(measured_data_df)
			measured_data_df = measured_data_df.set_index('n_trigger')
			measured_data_df['accepted'] = df['accepted']
			measured_data_df = measured_data_df.reset_index()
		else: # If there was no cleaning, we just accept all triggers...
			measured_data_df['accepted'] = True
		measured_data_df = measured_data_df.query('accepted==True').copy() # From now on we drop all useless data.
		
		if len(set(measured_data_df['device_name'])) == 2:
			devices_names = list(set(measured_data_df['device_name']))
		else:
			raise RuntimeError(f'A time resolution calculation requires two devices, but this beta scan has {len(set(measured_data_df["device_name"]))} device/s. Dont know how to handle this, sorry dude...')
		
		k1k2_device_names_df = {'device_name': [], 'device_number': []}
		for idx,device_name in enumerate(devices_names): # This is so I can use the same framework as in the TCT where there is only one detector but two pulses.
			device_number = idx+1
			measured_data_df.loc[measured_data_df['device_name']==device_name,'n_pulse'] = device_number
			k1k2_device_names_df['device_name'].append(device_name)
			k1k2_device_names_df['device_number'].append(device_number)
		k1k2_device_names_df = pandas.DataFrame(k1k2_device_names_df)
		k1k2_device_names_df.to_csv(Norberto.path_to_default_output_directory/Path('device_names_and_k1k2.csv'), index=False)

		final_results_data = []
		bootstrapped_replicas_data = []
		for k_bootstrap in range(N_BOOTSTRAP+1):

			bootstrapped_iteration = False
			if k_bootstrap > 0:
				bootstrapped_iteration = True

			if bootstrapped_iteration == False:
				data_df = measured_data_df.copy()
			else:
				data_df = resample_measured_data(measured_data_df)

			Delta_t_df = calculate_Delta_t_df(data_df)
			Delta_t_fluctuations_df = calculate_Delta_t_fluctuations_df(Delta_t_df)
			best_k1k2 = find_best_k1_k2(Delta_t_fluctuations_df)
			fitted_mu, fitted_sigma, fitted_amplitude = fit_gaussian_to_samples(Delta_t_df.set_index(['k_1 (%)','k_2 (%)']).loc[best_k1k2,'Δt (s)'])
			std = Delta_t_df.set_index(['k_1 (%)','k_2 (%)']).loc[best_k1k2,'Δt (s)'].std()

			bootstrapped_replicas_data.append(
				{
					'Δt k_MAD (s)': Delta_t_fluctuations_df.loc[best_k1k2,'k MAD(Δt) (s)'],
					'Δt std (s)': std,
					'Δt sigma from Gaussian fit (s)': fitted_sigma,
					'k_1 (%)': best_k1k2[0],
					'k_2 (%)': best_k1k2[1],
				}
			)
			if bootstrapped_iteration == True:
				continue

			# If we are here it is because we are not in a bootstrap iteration, it is the first iteration that is with the actual data.
			final_results_data.append(
				{
					'Δt k_MAD (s)': Delta_t_fluctuations_df.loc[best_k1k2,'k MAD(Δt) (s)'],
					'Δt std (s)': std,
					'Δt sigma from Gaussian fit (s)': fitted_sigma,
					'k_1 (%)': best_k1k2[0],
					'k_2 (%)': best_k1k2[1],
					'type': 'estimator value on the data',
				}
			)

			fig = plot_cfd(Delta_t_fluctuations_df)
			fig.update_layout(
				title = f'Time resolution vs CFD thresholds<br><sup>Measurement: {Norberto.measurement_name}</sup>'
			)
			fig.write_html(str(Norberto.path_to_default_output_directory/Path(f'CFD_plot.html')), include_plotlyjs = 'cdn')

			fig = go.Figure()
			fig.update_layout(
				yaxis_title = 'Number of events',
				xaxis_title = 'Δt (s)',
				title = f'Δt for k1={best_k1k2[0]}, k2={best_k1k2[1]}<br><sup>Measurement: {Norberto.measurement_name}</sup>'
			)
			samples_for_plot = np.array(list(Delta_t_df.loc[(Delta_t_df['k_1 (%)']==best_k1k2[0])&(Delta_t_df['k_2 (%)']==best_k1k2[1]),'Δt (s)']))
			fig.add_trace(
				plotly_utils.scatter_histogram(
					samples = samples_for_plot,
					name = f'Measured data',
					error_y = dict(type='auto'),
				)
			)
			x_axis_values = sorted(list(np.linspace(min(samples_for_plot),max(samples_for_plot),99)) + list(np.linspace(fitted_mu-5*fitted_sigma,fitted_mu+5*fitted_sigma,99)))
			fig.add_trace(
				go.Scatter(
					x = x_axis_values,
					y = gaussian(x_axis_values, fitted_mu, fitted_sigma, fitted_amplitude),
					name = f'Fitted Gaussian (σ={fitted_sigma*1e12:.2f} ps)',
				)
			)
			fig.write_html(
				str(Norberto.path_to_default_output_directory/Path(f'histogram k1 {best_k1k2[0]} k2 {best_k1k2[1]}.html')),
				include_plotlyjs = 'cdn',
			)

		bootstrapped_replicas_df = pandas.DataFrame.from_records(bootstrapped_replicas_data)

		bootstrapped_replicas_df[['k_1 (%)','k_2 (%)']] = bootstrapped_replicas_df[['k_1 (%)','k_2 (%)']].astype(int)

		bootstrapped_replicas_df_file_path = Norberto.path_to_default_output_directory/Path('bootstrap_results.csv')
		if bootstrapped_replicas_df_file_path.is_file():
			bootstrapped_replicas_df = pandas.concat(
				[
					bootstrapped_replicas_df,
					pandas.read_csv(bootstrapped_replicas_df_file_path)
				],
				ignore_index = True,
			)
		bootstrapped_replicas_df.to_csv(bootstrapped_replicas_df_file_path, index=False)

		stuff_to_append = dict(bootstrapped_replicas_df.std())
		stuff_to_append['type'] = 'std of the bootstrapped replicas'
		final_results_data.append(stuff_to_append)
		final_results_df = pandas.DataFrame.from_records(final_results_data)

		fig = go.Figure()
		fig.update_layout(
			title = f'Bootstrap replicas of Δt k_MAD<br><sup>Measurement: {Norberto.measurement_name}</sup>',
			xaxis_title = 'Estimation of σ (s)',
			yaxis_title = 'Number of events',
		)
		for stuff in {'Δt k_MAD','Δt sigma from Gaussian fit','Δt std'}:
			fig.add_trace(
				plotly_utils.scatter_histogram(
					samples = bootstrapped_replicas_df[f'{stuff} (s)'],
					name = stuff,
					error_y = dict(type='auto'),
				)
			)
		fig.write_html(
			str(Norberto.path_to_default_output_directory/Path(f'histogram bootstrap.html')),
			include_plotlyjs = 'cdn',
		)

		final_results_df.to_csv(Norberto.path_to_default_output_directory/Path('results.csv'), index=False)

def process_measurement_sweeping_voltage(directory:Path, force:bool=False, force_submeasurements:bool=False):
	Mariano = SmarterBureaucrat(directory, _locals=locals())
	
	Mariano.check_required_scripts_were_run_before('beta_scan_sweeping_bias_voltage.py')
	
	if force == False and Mariano.script_was_applied_without_errors(): # If this was already done, don't do it again...
		return
	
	with Mariano.do_your_magic():
		if Mariano.script_was_applied_without_errors('beta_scan_sweeping_bias_voltage.py'): # This means that we are dealing with a voltage scan that should contain submeasurements at each voltage.
			Mariano.check_required_scripts_were_run_before('beta_scan_sweeping_bias_voltage.py')
			
			submeasurements_dict = Mariano.find_submeasurements('beta_scan_sweeping_bias_voltage.py')
			if submeasurements_dict is None:
				raise RuntimeError(f'I was expecting to find submeasurements in {Mariano.path_to_output_directory_of_script_named("beta_scan_sweeping_bias_voltage.py")}, but I cant...s')
			
			time_resolution_data = []
			for measurement_name,path_to_submeasurement in submeasurements_dict.items():
				process_single_voltage_measurement(path_to_submeasurement, force=force_submeasurements) # Recursive call...
				Teutónio = SmarterBureaucrat(
					path_to_submeasurement,
					_locals = locals(),
				)
				df = pandas.read_csv(Teutónio.path_to_output_directory_of_script_named('time_resolution_beta_scan.py')/Path('results.csv'))
				bootstrap_df = pandas.read_csv(Teutónio.path_to_output_directory_of_script_named('time_resolution_beta_scan.py')/Path('bootstrap_results.csv'))
				this_measurement_error = bootstrap_df['Δt sigma from Gaussian fit (s)'].std()
				time_resolution_data.append(
					{
						'Δt sigma from Gaussian fit (s)': float(list(df.query('type=="estimator value on the data"')['Δt sigma from Gaussian fit (s)'])[0]),
						'Δt sigma from Gaussian fit (s) bootstrapped error estimation': this_measurement_error,
						'Measurement name': measurement_name,
						'Bias voltage (V)': int(get_voltage_from_measurement(measurement_name)[:-1]),
					}
				)

			time_resolution_df = pandas.DataFrame.from_records(time_resolution_data)

			df = time_resolution_df.sort_values(by='Bias voltage (V)')
			fig = plotly_utils.line(
				title = f'Measured jitter vs bias voltage with beta source<br><sup>Measurement: {Mariano.measurement_name}</sup>',
				data_frame = df,
				x = 'Bias voltage (V)',
				y = 'Δt sigma from Gaussian fit (s)',
				error_y = 'Δt sigma from Gaussian fit (s) bootstrapped error estimation',
				hover_data = ['Measurement name'],
				markers = 'circle',
				labels = {
					'Δt sigma from Gaussian fit (s)': '√(σ<sub>1</sub>²+σ<sub>2</sub>²) (s)',
				}
			)
			fig.write_html(
				str(Mariano.path_to_default_output_directory/Path('jitter vs bias voltage.html')),
				include_plotlyjs = 'cdn',
			)
			time_resolution_df.to_csv(Mariano.path_to_default_output_directory/Path('time_resolution_vs_bias_voltage.csv'))
			
			return

def script_core(directory:Path, force:bool=False, force_submeasurements:bool=False):
	John = SmarterBureaucrat(
		directory,
		_locals = locals(),
	)
	
	if John.script_was_applied_without_errors('beta_scan_sweeping_bias_voltage.py'):
		process_measurement_sweeping_voltage(directory, force, force_submeasurements)
	elif John.script_was_applied_without_errors('beta_scan.py'):
		process_single_voltage_measurement(directory, force)
	else:
		raise RuntimeError(f'Dont know how to process {directory}...')

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
		force = True,
		force_submeasurements = False,
	)
