from pathlib import Path
import pandas
from bureaucrat.Bureaucrat import Bureaucrat
import utils

MEASUREMENTS_DATA_PATH = Path('../../measurements_data')

def measurement_type(measurement_name: str) -> str:
	"""Returns the type of measurement (beta scan, TCT scan, IV, etc.).
	
	Example
	-------
	```
	MEASUREMENTS = {
		'20220404021350_MS07_1DScan_228V',
		'20220404021350_MS07_1DScan_228V',
		'20220403214116_MS07_sweeping_bias_voltage',
		'20220328170454_MS03_IV_Curve',
		'20220405203845_MS08_3_probe_station',
		'20220317155531_BetaScan_SpeedyGonzalez12_at_98V',
	}
	for m in MEASUREMENTS:
		print(f'{m} | {measurement_type(m)}')
	```
	produces
	```
	20220328170454_MS03_IV_Curve | IV curve
	20220404021350_MS07_1DScan_228V | TCT 1D scan fixed voltage
	20220317155531_BetaScan_SpeedyGonzalez12_at_98V | beta fixed voltage
	20220405203845_MS08_3_probe_station | IV curve probe station
	20220403214116_MS07_sweeping_bias_voltage | TCT 1D scan sweeping bias voltage
	```
	"""
	measurement_type = 'unknown'
	sub_directories_in_this_measurement = [p.parts[-1] for p in (MEASUREMENTS_DATA_PATH/Path(measurement_name)).iterdir()]
	if 'beta' in measurement_name.lower():
		measurement_type = 'beta'
		if 'sweeping' not in measurement_name.lower():
			measurement_type += ' fixed voltage'
		elif 'sweeping_bias_voltage' in measurement_name.lower():
			measurement_type += ' voltage scan'
	elif 'iv' in measurement_name.lower() and 'curve' in measurement_name.lower():
		measurement_type = 'IV curve'
	if all([s in measurement_name.lower() for s in {'probe','station'}]):
		measurement_type = 'IV curve probe station'
	elif '1DScan' in measurement_name and 'scan_1D' in sub_directories_in_this_measurement:
		measurement_type = 'TCT 1D scan fixed voltage'
	elif 'sweeping_bias_voltage' in measurement_name.lower() and 'scan_1D_sweeping_bias_voltage' in sub_directories_in_this_measurement:
		measurement_type = 'TCT 1D scan sweeping bias voltage'
	return measurement_type

def read_and_preprocess_measurement_TCT_1DScan_fixed_voltage(measurement_name: str):
	"""Reads the "main data" from a TCT measurement and does some pre-processing
	so it is easier and more standard in the top level scripts to handle
	it.
	
	Returns
	-------
	data_df: pandas.DataFrame
		Data frame with the data.
	"""
	data_df = pandas.read_feather(MEASUREMENTS_DATA_PATH/Path(measurement_name)/Path('parse_waveforms_from_scan')/Path('data.fd'))
	return data_df
		
class MeasurementHandler:
	"""A class specialized in handling the measurements of the ETL
	Market Survey.
	"""
	def __init__(self, measurement_name: str):
		"""Creates a `MeasurementHantler`.
		
		Parameters
		----------
		measurement_name: str
			Name (not path) of a measurement.
		"""
		if not isinstance(measurement_name, str):
			raise TypeError(f'`measurement_name` must be an instance of {type("hola")}, received object of type {type(measurement_name)}.')
		self._measurement_name = measurement_name
	
	@property
	def measurement_name(self) -> str:
		return self._measurement_name
	
	@property
	def measurement_type(self) -> str:
		"""Returns the measurement type, e.g. "TCT 1D scan sweeping bias voltage"
		or "IV curve probe station". To see all possible options refer
		to the source code.
		"""
		if not hasattr(self, '_measurement_type'):
			self._measurement_type = measurement_type(self.measurement_name)
		return self._measurement_type
	
	@property
	def measurement_data(self) -> pandas.DataFrame:
		"""Reads the "main data" from this measurement and does some pre-processing
		so it is easier and more standard in the top level scripts to handle
		it.
		
		Returns
		-------
		data_df: pandas.DataFrame
			Data frame with the data.
		"""
		if not hasattr(self, '_measurement_data_df'):
			if self.measurement_type == 'TCT 1D scan fixed voltage':
				self._measurement_data_df = read_and_preprocess_measurement_TCT_1DScan_fixed_voltage(self.measurement_name)
			else:
				raise NotImplementedError(f"Don't know how to read a measurement of type {repr(self.measurement_type)}.")
		return self._measurement_data_df
	
	def tag_left_and_right_pads(self) -> None:
		"""If the measurement is one in which we can define which is the
		left and right pads (e.g. a TCT 1D scan), it adds a column to the
		measured data with such tag. Otherwise, a `TypeError` is rised.
		"""
		if self.measurement_type == 'TCT 1D scan fixed voltage':
			if 'Pad' not in self.measurement_data.columns:
				pads_df = utils.tag_left_right_pad(self.measurement_data)
				self.measurement_data['Pad'] = pads_df['Pad']
		else:
			raise TypeError(f"Don't know how to tag left and right pads for a measurement of type {repr(self.measurement_type)}.")
	
if __name__ == '__main__':
	MEASUREMENTS = {
		'20220404021350_MS07_1DScan_228V',
		'20220404021350_MS07_1DScan_228V',
		# ~ '20220403214116_MS07_sweeping_bias_voltage',
		# ~ '20220328170454_MS03_IV_Curve',
		# ~ '20220405203845_MS08_3_probe_station',
		# ~ '20220317155531_BetaScan_SpeedyGonzalez12_at_98V',
	}
	for m in MEASUREMENTS:
		handler = MeasurementHandler(m)
		print(handler.measurement_name)
		print(handler.measurement_type)
		handler.tag_left_and_right_pads()
		print(handler.measurement_data)
