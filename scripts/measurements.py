from pathlib import Path

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

if __name__ == '__main__':
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
