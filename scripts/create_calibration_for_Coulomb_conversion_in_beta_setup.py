from bureaucrat.Bureaucrat import Bureaucrat # https://github.com/SengerM/bureaucrat
from pathlib import Path
import pandas
import measurements
import csv
import grafica.plotly_utils.utils as plotly_utils # https://github.com/SengerM/grafica
from devices_info import devices_info_df
import numpy as np
from scipy.constants import elementary_charge

def script_core(beta_scans_on_PINs:list, PINs_names:list):
	"""Creates charge calibration data to convert from `V*s` to `Coulomb`
	using data from beta scans performed on PIN devices.
	
	Parameters
	----------
	beta_scans_on_PINs: list of Path
		A list of `Path`s pointing to the base directory of one or more
		beta scans performed on PIN diodes.
	PINs_names: list of str
		A list with the names of the PINs in each of the beta scans. This
		is because a beta scan is typically performed between at least
		two devices, so I must know which is the PIN in each.
	"""
	KalEl = Bureaucrat(
		measurements.MEASUREMENTS_DATA_PATH/Path('Coulomb_calibration'),
        new_measurement = True,
		variables = locals(),
	)

	with KalEl.verify_no_errors_context():
		measured_collected_charge_df = pandas.DataFrame()
		for PIN_beta_scan_path,PIN_name in zip(beta_scans_on_PINs,PINs_names):
			if not (PIN_beta_scan_path/Path('collected_charge_vs_bias_voltage_beta_scan/.script_successfully_applied')).is_file():
				raise RuntimeError(f'I need the "collected charge vs bias voltage" data, I cannot find it for measurement {PIN_beta_scan_path.parts[-1]}')
			df = pandas.read_csv(PIN_beta_scan_path/Path('collected_charge_vs_bias_voltage_beta_scan/collected_charge_vs_bias_voltage.csv'))
			df['Measurement name'] = PIN_beta_scan_path.parts[-1]
			measured_collected_charge_df = pandas.concat(
				[measured_collected_charge_df, df.query(f'`Device name`=="{PIN_name}"')],
				ignore_index = True,
			)
		
		collected_charge_single_values_df = measured_collected_charge_df.query('`Bias voltage (V)`>=100').groupby('Device name').agg({'Collected charge (V s) x_mpv value_on_data': [np.mean, np.std]})
		
		thickness = 50e-6 # Hardcoded here because it is the same for all the FBK devices.
		PIN_charge_in_theory_in_Coulomb = elementary_charge*(31*np.log(thickness/1e-6)+128)*thickness/1e-6/3.65 # https://sengerm.github.io/html-github-hosting/210721_Commissioning_of_Chubut_board/210721_Commissioning_of_Chubut_board.html
		conversion_factor = PIN_charge_in_theory_in_Coulomb/collected_charge_single_values_df['Collected charge (V s) x_mpv value_on_data'].mean()
		conversion_factor.to_csv(KalEl.processed_data_dir_path/Path('conversion_factor (Coulomb over Volt over second).csv'))
		
		for col in measured_collected_charge_df.columns:
			if '(V s)' in col:
				measured_collected_charge_df[col.replace('(V s)','(C)')] = measured_collected_charge_df[col]*conversion_factor['mean']
				if 'std' in col:
					measured_collected_charge_df[col.replace('(V s)','(C)')] = (measured_collected_charge_df[col.replace('(V s)','(C)')]**2 + conversion_factor['std']**2)**.5
		
		fig = plotly_utils.line(
			title = f'Collected charge vs bias voltage with beta source<br><sup>Measurement: {KalEl.measurement_name}</sup>',
			data_frame = measured_collected_charge_df.sort_values(by=['Bias voltage (V)']),
			x = 'Bias voltage (V)',
			y = 'Collected charge (V s) x_mpv value_on_data',
			error_y = 'Collected charge (V s) x_mpv std',
			hover_data = sorted(measured_collected_charge_df),
			markers = 'circle',
			labels = {
				'Collected charge (V s) x_mpv value_on_data': 'Collected charge (V s)',
			},
			color = 'Device name',
		)
		for device_name in collected_charge_single_values_df.index:
			fig.add_hline(
				y = collected_charge_single_values_df.loc[device_name, ('Collected charge (V s) x_mpv value_on_data','mean')],
				annotation_text = device_name,
			)
		fig.write_html(
			str(KalEl.processed_data_dir_path/Path('collected charge vs bias voltage.html')),
			include_plotlyjs = 'cdn',
		)
		
if __name__ == '__main__':
	
	a = {
		"20220407154640_BetaScan_MS11_PIN_sweeping_bias_voltage": 'MS11_PIN',
		"20220412121103_BetaScan_MS12_PIN_sweeping_bias_voltage": 'MS12_PIN',
		'20220428185145_BetaScan_MS40PIN_sweeping_bias_voltage': 'MS40PIN',
		'20220427174812_BetaScan_MS42PIN_sweeping_bias_voltage': 'MS42PIN',
	}
	script_core(
		beta_scans_on_PINs = [measurements.MEASUREMENTS_DATA_PATH/Path(m) for m in a.keys()],
		PINs_names = [a[k] for k in a.keys()],
	)
