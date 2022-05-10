from math import nan
from bureaucrat.Bureaucrat import Bureaucrat # https://github.com/SengerM/bureaucrat
from pathlib import Path
import pandas
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import InterpolatedUnivariateSpline #from scipy.interpolate import interp1d
from scipy.misc import derivative
from landaupy import langauss
from landaupy import landau
from scipy.stats import median_abs_deviation
from scipy.optimize import curve_fit
from grafica.plotly_utils.utils import scatter_histogram # https://github.com/SengerM/grafica
from plotly.subplots import make_subplots

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import scipy.cluster.hierarchy as sch

from utils import clean_data

SET_OF_COLUMNS_TO_IGNORE = {'n_waveform','n_trigger','When','device_name','Accepted'}

def hex_to_rgba(h, alpha):
    return tuple([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] + [alpha])

def apply_cuts(data_df, cuts_df):
	"""
	Given a dataframe `cuts_df` with one cut per row, e.g.
	```
				  variable  device_name cut type     cut value
				  t_50 (s)            1    lower  1.341500e-07
				  t_50 (s)            1   higher  1.348313e-07
	Collected charge (V s)            4    lower  2.204645e-11

	```
	this function returns a series with the index `n_trigger` and the value
	either `True` or `False` stating if such trigger satisfies ALL the
	cuts at the same time. For example using the previous example a
	trigger with charge 3e-12 and t_50 6.45e-8 will be `True` but if any
	of the variables in any of the channels is outside the range, it will
	be `False`.
	"""
	for device_name in set(cuts_df['device_name']):
		if device_name not in set(data_df['device_name']):
			raise ValueError(f'There is a cut in `device_name={repr(device_name)}` but the measured data does not contain this device, measured devices are {set(data_df["device_name"])}.')
	data_df = data_df.pivot(
		index = 'n_trigger',
		columns = 'device_name',
		values = list(set(data_df.columns) - {'device_name'}),
	)
	triggers_accepted_df = pandas.DataFrame({'accepted': True}, index=data_df.index)
	for idx, cut_row in cuts_df.iterrows():
		if cut_row['cut type'] == 'lower':
			triggers_accepted_df['accepted'] &= data_df[(cut_row['variable'],cut_row['device_name'])] >= cut_row['cut value']
		elif cut_row['cut type'] == 'higher':
			triggers_accepted_df['accepted'] &= data_df[(cut_row['variable'],cut_row['device_name'])] <= cut_row['cut value']
		else:
			raise ValueError('Received a cut of type `cut type={}`, dont know that that is...'.format(cut_row['cut_type']))
	return triggers_accepted_df

def binned_fit_langauss(samples, bins='auto', nan='remove'):
	if nan == 'remove':
		samples = samples[~np.isnan(samples)]
	hist, bin_edges = np.histogram(samples, bins, density=True)
	bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2
	# Add an extra bin to the left:
	hist = np.insert(hist, 0, sum(samples<bin_edges[0]))
	bin_centers = np.insert(bin_centers, 0, bin_centers[0]-np.diff(bin_edges)[0])
	# Add an extra bin to the right:
	hist = np.append(hist,sum(samples>bin_edges[-1]))
	bin_centers = np.append(bin_centers, bin_centers[-1]+np.diff(bin_edges)[0])
	landau_x_mpv_guess = bin_centers[np.argmax(hist)]
	landau_xi_guess = median_abs_deviation(samples)/5
	gauss_sigma_guess = landau_xi_guess/10
	popt, pcov = curve_fit(
		lambda x, mpv, xi, sigma: langauss.pdf(x, mpv, xi, sigma),
		xdata = bin_centers,
		ydata = hist,
		p0 = [landau_x_mpv_guess, landau_xi_guess, gauss_sigma_guess],
		# ~ bounds = ([0]*3, [float('inf')]*3), # Don't know why setting the limits make this to fail.
	)
	return popt, pcov, hist, bin_centers

def apply_mva(data_df, plot_path, max_pca_components = 6, max_kmeans_clusters = 12, ignore_columns = []):
	new_df=[]
	new_columns=[]
	cluster_columns=[]

	unused_columns = ignore_columns

	for device_name in set(data_df['device_name']):
		# Try PCA
		print("Applying Multivariate Analysis to {} data".format(device_name))
		device_data_df = data_df.loc[data_df['device_name'] == device_name].drop(
			unused_columns,
			axis=1,
		)
		device_sc = StandardScaler()
		device_sc.fit(device_data_df)
		device_pca = PCA(n_components=max_pca_components) # Immediately do the max components because if we want less, we can just ignore the higher components
		device_pca.fit(device_sc.transform(device_data_df))
		print("  The PCA explains the variance to: {}".format(device_pca.explained_variance_ratio_))

		tmp_data = device_pca.transform(device_sc.transform(data_df.drop(unused_columns,axis=1)))
		pca_columns = []
		for column in range(max_pca_components):
			pca_columns += ["PC{} {}".format(column+1, device_name)]
		new_df += [pandas.DataFrame(tmp_data, columns=pca_columns)]
		new_columns += pca_columns

		# Try k-means clustering
		distortions = []
		silhouette_scores = []
		K = range(1,max_kmeans_clusters)
		for cluster_size in K:
			kmeans = cluster.KMeans(n_clusters=cluster_size, init='k-means++', random_state=200)
			kmeans = kmeans.fit(device_data_df)
			distortions.append(kmeans.inertia_)
			if cluster_size > 1:
				labels = kmeans.labels_
				silhouette_score = metrics.silhouette_score(
					device_data_df,
					labels,
					metric='euclidean',
					#sample_size=1000,
					random_state=200,
				)
				silhouette_scores.append(silhouette_score)
			else:
				silhouette_scores.append(np.nan)
			identified_clusters = kmeans.predict(data_df.drop(unused_columns,axis=1))
			column = "k-means {} Cluster {}".format(cluster_size, device_name)
			new_df += [pandas.DataFrame(identified_clusters, columns=[column])]
			cluster_columns += [column]
			new_columns += [column]
		df = pandas.DataFrame({'Clusters': K, 'Distortions': distortions, 'Silhouette Score': silhouette_scores})
		fig = make_subplots(specs=[[{"secondary_y": True}]])
		fig.add_trace(
		    go.Scatter(x=df['Clusters'], y=df['Distortions'], name="Distortions"),
			secondary_y=False,
		)
		fig.add_trace(
		    go.Scatter(x=df['Clusters'], y=df['Silhouette Score'], name='Silhouette Score'),
			secondary_y=True,
		)
		fig.update_layout(
			title_text="Cluster number selection for {}".format(device_name)
		)
		fig.update_xaxes(title_text="Clusters")
		fig.update_yaxes(title_text='Distortions', secondary_y=False)
		fig.update_yaxes(title_text='Silhouette Score', secondary_y=True)
		(plot_path/Path('kMeans')).mkdir(exist_ok=True, parents=True)
		fig.write_html(
			str(plot_path/Path('kMeans/selection_clusters_{}.html'.format(device_name))),
			include_plotlyjs = 'cdn',
		)

		# Try Hierarchical Clustering (the plots from this are slow to display)
		fig = ff.create_dendrogram(device_data_df.astype(float),
								linkagefun = lambda x: sch.linkage(x, "ward"),)
		fig.update_layout(title = 'Hierarchical Clustering for {}'.format(device_name),
							yaxis_title='Euclidean Distance')
		(plot_path/Path('Hierarchical_Clustering')).mkdir(exist_ok=True, parents=True)
		fig.write_html(
			str(plot_path/Path('Hierarchical_Clustering/{}.html'.format(device_name))),
			include_plotlyjs = 'cdn',
		)

		# Try Affinity Propagation (does not work well with this dataset, probably needs tuning)
		#ap = cluster.AffinityPropagation(random_state = 100)
		#ap = ap.fit(device_data_df)
		#identified_clusters = ap.predict(measured_data_df.drop(unused_columns,axis=1))
		#new_df += [pandas.DataFrame(identified_clusters, columns=["Affinity Propagation Cluster {}".format(device_name)])]
		#cluster_columns += ["Affinity Propagation Cluster {}".format(device_name)]

	# Add the new data columns to the dataframe
	for df in new_df:
		data_df = data_df.join(df)

	return (data_df, new_columns)

def update_scatter_plot_selection(fig):
	for k in range(len(fig.data)):
		fig.data[k].update(
			selected = dict(
				marker = dict(
					opacity = 1,
					color = 'black',
				)
			),
			# ~ unselected = dict(
				# ~ marker = dict(
					# ~ opacity = 0.01
				# ~ )
			# ~ ),
		)

def plot_mva(data_df, plot_path, max_pca_components = 6, measurement_name="", scatter_columns=[], selection_column = None):
	# Plot first two components of PCA for both devices
	pca_scatter_columns = scatter_columns
	(plot_path/Path('PCA')).mkdir(exist_ok=True, parents=True)
	for device_name in set(data_df['device_name']):
		scatter_opt = {
			'dimensions': sorted(set(["PC1 {}".format(device_name), "PC2 {}".format(device_name)] + pca_scatter_columns)),
			'title': "PCA of {} scatter matrix plot<br><sup>Measurement: {}</sup>".format(device_name, measurement_name),
			'symbol': 'device_name',
			'hover_data': ['n_trigger'],
		}
		if selection_column is not None:
			scatter_opt['color'] = selection_column
			scatter_opt['color_discrete_map'] = {False: 'red', True: 'green'}
			scatter_opt['symbol_map'] = {True: 'circle', False: 'x'}
		fig = px.scatter_matrix(
			data_df,
			**scatter_opt
		)
		fig.update_traces(diagonal_visible=False, showupperhalf=False)
		update_scatter_plot_selection(fig)
		if selection_column is None:
			fig.write_html(
				str(plot_path/Path('PCA/scatter_matrix_{}.html'.format(device_name))),
				include_plotlyjs = 'cdn',
			)
		else:
			fig.write_html(
				str(plot_path/Path('PCA/selected_scatter_matrix_{}.html'.format(device_name))),
				include_plotlyjs = 'cdn',
			)

	# Plot PCA correlation
	pca_correlation_columns = []
	max_pca_correlation = min(4,max_pca_components)
	for device_name in set(data_df['device_name']):
		for component in range(max_pca_correlation):
			pca_correlation_columns += ["PC{} {}".format(component+1, device_name)]
	scatter_opt = {
		'dimensions': sorted(set(pca_correlation_columns)),
		'title': "PCA correlation plot<br><sup>Measurement: {}</sup>".format(measurement_name),
		'symbol': 'device_name',
		'hover_data': ['n_trigger'],
	}
	if selection_column is not None:
		scatter_opt['color'] = selection_column
		scatter_opt['color_discrete_map'] = {False: 'red', True: 'green'}
		scatter_opt['symbol_map'] = {True: 'circle', False: 'x'}
	fig = px.scatter_matrix(
		data_df,
		**scatter_opt
	)
	fig.update_traces(diagonal_visible=False, showupperhalf=False)
	update_scatter_plot_selection(fig)
	if selection_column is None:
		fig.write_html(
			str(plot_path/Path('PCA/correlation.html')),
			include_plotlyjs = 'cdn',
		)
	else:
		fig.write_html(
			str(plot_path/Path('PCA/selected_correlation.html')),
			include_plotlyjs = 'cdn',
		)

def script_core(directory: Path, plot_waveforms=False):
	John = Bureaucrat(
		directory,
		variables = locals(),
	)

	plots_dir_path = John.processed_data_dir_path/Path('plots')
	plots_dir_path.mkdir(exist_ok=True, parents=True)
	mva_dir_path = John.processed_data_dir_path/Path('MVA')
	mva_dir_path.mkdir(exist_ok=True, parents=True)

	cuts_file_path = John.measurement_base_path/Path('cuts.csv')

	try:
		measured_data_df = pandas.read_feather(John.processed_by_script_dir_path('beta_scan.py')/Path('measured_data.fd'))
	except FileNotFoundError:
		measured_data_df = pandas.read_csv(John.processed_by_script_dir_path('beta_scan.py')/Path('measured_data.csv'))

	with John.verify_no_errors_context():
		# TODO: Maybe we need to add this line to remove NaN to the other scripts
		measured_data_df = clean_data(measured_data_df)

		measured_data_df, new_columns = apply_mva(measured_data_df, mva_dir_path, ignore_columns=[
			'Temperature (°C)',
			'device_name',
			'Humidity (%RH)',
			'Bias current (A)',
			'When',
			'Bias voltage (V)',
			'n_trigger',
		])

		# Make MVA plots before selection cuts
		plot_mva(measured_data_df, mva_dir_path,
			measurement_name=John.measurement_name,
			scatter_columns=[
				"Time over noise (s)",
				"Amplitude (V)",
				"Collected charge (V s)",
				"t_50 (s)",
				"Peak start time (s)",
				"Rise time (s)"
			]
		)

		try:
			cuts_df = pandas.read_csv(cuts_file_path)
			cuts_df.to_csv(John.processed_data_dir_path/Path(f'cuts.csv'))

			filtered_triggers_df = apply_cuts(measured_data_df, cuts_df)
			filtered_triggers_df.reset_index().to_feather(John.processed_data_dir_path/Path('clean_triggers.fd'))
		except FileNotFoundError:
			print(f'Cannot find `{cuts_file_path}` specifying the cuts, will accept all triggers.')
			cuts_df = pandas.DataFrame(columns=['variable']) # Create dummy df.
		measured_data_df = measured_data_df.set_index('n_trigger')
		try:
			measured_data_df['Accepted'] = filtered_triggers_df
		except NameError:
			measured_data_df['Accepted'] = True # Accept all triggers.
		measured_data_df = measured_data_df.reset_index()


		for column in measured_data_df:
			if column in SET_OF_COLUMNS_TO_IGNORE:
				continue
			histogram_fig = px.histogram(
				measured_data_df,
				x = column,
				facet_col = 'device_name',
				title = f'{column}<br><sup>Measurement: {John.measurement_name}</sup>',
				color = 'Accepted',
				color_discrete_map = {False: 'red', True: 'green'},
				pattern_shape_map = {False: 'x', True: ''},
				marginal = 'rug',
				hover_data = ['n_trigger'],
			)
			if 'collected charge' in column.lower(): # LANGAUSS FIT!
				fig = go.Figure()
				fig.update_layout(
					title = f'Langauss fit on "accepted events"<br><sup>Measurement: {John.measurement_name}</sup>',
					xaxis_title = column,
					yaxis_title = 'Probability density',
				)
				colors = iter(px.colors.qualitative.Plotly)
				for device_name in sorted(set(measured_data_df['device_name'])):
					_samples = measured_data_df.query('Accepted==True').query(f'device_name=={repr(device_name)}')['Collected charge (V s)']
					popt, _, hist, bin_centers = binned_fit_langauss(_samples, nan="")
					this_channel_color = next(colors)

					fig.add_trace(
						scatter_histogram(
							samples = _samples,
							error_y = dict(type='auto'),
							density = True,
							name = f'Data {device_name}',
							line = dict(color = this_channel_color),
							legendgroup = device_name,
							nan_policy="",
						)
					)
					x_axis = np.linspace(min(bin_centers),max(bin_centers),999)
					fig.add_trace(
						go.Scatter(
							x = x_axis,
							y = langauss.pdf(x_axis, *popt),
							name = f'Langauss fit {device_name}<br>x<sub>MPV</sub>={popt[0]:.2e}<br>ξ={popt[1]:.2e}<br>σ={popt[2]:.2e}',
							line = dict(color = this_channel_color, dash='dash'),
							legendgroup = device_name,
						)
					)
					fig.add_trace(
						go.Scatter(
							x = x_axis,
							y = landau.pdf(x_axis, popt[0], popt[1]),
							name = f'Landau component {device_name}',
							line = dict(color = f'rgba{hex_to_rgba(this_channel_color, .3)}', dash='dashdot'),
							legendgroup = device_name,
						)
					)
				fig.write_html(
					str(John.processed_data_dir_path/Path(f'{column} langauss fit.html')),
					include_plotlyjs = 'cdn',
				)
			if column in set(cuts_df['variable']) or column in {'Collected charge (V s)'}:
				ecdf_fig = px.ecdf(
					measured_data_df,
					x = column,
					color = 'device_name',
					title = f'{column}<br><sup>Measurement: {John.measurement_name}</sup>',
					marginal = 'histogram',
					facet_row = 'Accepted',
					hover_data = ['n_trigger'],
				)
				cuts_to_draw_df = cuts_df.loc[cuts_df['variable']==column]
				if len(cuts_to_draw_df) > 0:
					for device_name in sorted(set(cuts_to_draw_df['device_name'])):
						for cut_type in sorted(set(cuts_to_draw_df.loc[cuts_to_draw_df['device_name']==device_name,'cut type'])):
							for fig in [histogram_fig, ecdf_fig]:
								fig.add_vline(
									x = float(cuts_df.loc[(cuts_df['device_name']==device_name)&(cuts_df['cut type']==cut_type)&(cuts_df['variable']==column), 'cut value']),
									opacity = .5,
									annotation_text = f'{device_name} {cut_type} cut️',
									line_color = 'black',
									line_dash = 'dash',
									annotation_textangle = -90,
									annotation_position = 'bottom left',
								)
				ecdf_fig.write_html(
					str(plots_dir_path/Path(f'{column} ECDF.html')),
					include_plotlyjs = 'cdn',
				)

			histogram_fig.write_html(
				str(plots_dir_path/Path(f'{column} histogram.html')),
				include_plotlyjs = 'cdn',
			)

		columns_for_scatter_matrix_plot = set(measured_data_df.columns) - SET_OF_COLUMNS_TO_IGNORE - {'Time over 20% (s)'} - {f't_{i*10} (s)' for i in [1,2,3,4,6,7,8,9]} - {'Humidity (%RH)','Temperature (°C)','Bias voltage (V)','Bias current (A)'} - set(new_columns)
		df = measured_data_df
		fig = px.scatter_matrix(
			df,
			dimensions = sorted(columns_for_scatter_matrix_plot),
			title = f'Scatter matrix plot<br><sup>Measurement: {John.measurement_name}</sup>',
			symbol = 'device_name',
			color = 'Accepted',
			color_discrete_map = {False: 'red', True: 'green'},
			symbol_map = {True: 'circle', False: 'x'},
			hover_data = ['n_trigger'],
		)
		fig.update_traces(diagonal_visible=False, showupperhalf=False)
		update_scatter_plot_selection(fig)
		fig.write_html(
			str(John.processed_data_dir_path/Path('scatter_matrix.html')),
			include_plotlyjs = 'cdn',
		)

		# Make MVA plots after selection cuts
		plot_mva(measured_data_df, mva_dir_path,
			measurement_name=John.measurement_name,
			scatter_columns=[
				"Time over noise (s)",
				"Amplitude (V)",
				"Collected charge (V s)",
				"t_50 (s)",
				"Peak start time (s)",
				"Rise time (s)"
			],
			selection_column = 'Accepted'
		)

	# Plot waveforms ---
	if plot_waveforms == True:
		try:
			waveforms_df = pandas.read_feather(John.processed_by_script_dir_path('beta_scan.py')/Path('waveforms.fd'))
		except FileNotFoundError:
			waveforms_df = pandas.read_csv(John.processed_by_script_dir_path('beta_scan.py')/Path('waveforms.csv'))

		waveforms_df = waveforms_df.set_index('n_trigger')
		waveforms_df['Accepted'] = True # Accept all triggers.
		try:
			waveforms_df['Accepted'] = filtered_triggers_df
		except NameError:
			waveforms_df['Accepted'] = True # Accept all triggers.
		waveforms_df = waveforms_df.reset_index()

		fig = px.line(
			waveforms_df,
			title = f'Waveforms<br><sup>Measurement: {John.measurement_name}</sup>',
			x = 'Time (s)',
			y = 'Amplitude (V)',
			facet_row = 'device_name',
			line_group = 'n_trigger',
			color = 'Accepted',
			color_discrete_map = {False: 'red', True: 'green'},
			render_mode = 'webgl', # https://plotly.com/python/webgl-vs-svg/
			category_orders = {'device_name': sorted(set(waveforms_df['device_name'])), 'Accepted': [True, False]},
		)
		fig.update_yaxes(matches=None)
		fig.update_traces(line=dict(width=1), opacity=.1)
		fig.write_html(
			str(John.processed_data_dir_path/Path('waveforms.html')),
			include_plotlyjs = 'cdn',
		)

		accepted_status_for_the_plot = True
		fig = make_subplots(
			rows = 2,
			cols = 1,
			shared_xaxes = True,
			vertical_spacing = 0.02,
		)
		fig.update_xaxes(title_text = 'Time (s)', row=2, col=1)
		for n_row in [1,2]:
			fig.update_yaxes(title_text = 'Amplitude (V)', row=n_row, col=1)
		fig.update_layout(
			title = f'Waveforms 2D histogram with `accepted={accepted_status_for_the_plot}`<br><sup>Measurement: {John.measurement_name}</sup>',
			legend = dict(yanchor="top", y=1, xanchor="right", x=1.3),
		)
		for device_idx,device_name in enumerate(sorted(set(waveforms_df['device_name']))):
				df = waveforms_df.query(f'device_name=={repr(device_name)}').query(f'Accepted=={accepted_status_for_the_plot}')
				fig.add_trace(
					go.Histogram2d(
						x = df['Time (s)'],
						y = df['Amplitude (V)'],
						xbins = dict(
							start = min(waveforms_df['Time (s)']),
							end = max(waveforms_df['Time (s)']),
							size = np.diff(sorted(set(waveforms_df['Time (s)'])))[0],
						),
						ybins = dict(
							start = min(df['Amplitude (V)']),
							end = max(df['Amplitude (V)']),
							size = np.diff(sorted(set(df['Amplitude (V)'])))[0],
						),
						colorscale=[[0, 'rgba(0,0,0,0)'], [0.00000001, '#000096'], [.05, '#9300a3'], [.25, '#ff9500'], [1, '#ffffff']],
						histnorm = 'probability',
						hoverinfo = 'skip',
					),
					row = device_idx+1,
					col = 1,
				)
				fig.add_annotation(
					showarrow = False,
					text = device_name,
					textangle = 90,
					x = 1,
					xanchor = 'left',
					xref = "paper",
					y = 1-(2*device_idx+1)/(2*len(set(waveforms_df['device_name']))),
					yanchor = 'middle',
					yref = "paper",
				)
		fig.update_traces(showscale=False)
		fig.write_html(
			str(John.processed_data_dir_path/Path('waveforms_histogram.html')),
			include_plotlyjs = 'cdn',
		)

########################################################################

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='Cleans a beta scan according to some criterion.')
	parser.add_argument('--dir',
		metavar = 'path',
		help = 'Path to the base measurement directory.',
		required = True,
		dest = 'directory',
		type = str,
	)
	parser.add_argument(
		'--plot-waveforms',
		help = 'If this flag is passed, all the waveforms are plotted. Default is not plot, reason is that it takes some time and the resulting plots are heavy.',
		action = 'store_true',
		dest = 'plot_waveforms',
	)

	args = parser.parse_args()
	script_core(Path(args.directory), plot_waveforms = args.plot_waveforms)
