import pandas

def read_devices_info_sheet():
	"""Reads the `devices.xlsx` file where all the info about each device
	is supposed to be stored and returns it as a `pandas.dataframe`.
	"""
	return pandas.read_excel('../../doc/devices.xlsx').set_index('device_name')

devices_info_df = read_devices_info_sheet() # This is here such that we can import this object and only read the file once.

def get_device_alias(device_name:str) -> str:
	"""Returns a string identifying the device in an insitute-cross way,
	e.g. `"FBK-W7-T9-GR3_0"`. This function should be used when producing
	plots to share with others, instead of the internal device names that
	we use at UZH.
	"""
	if device_name not in devices_info_df.index:
		raise ValueError(f'`device_name` must be one of {sorted(devices_info_df.index)}, received {repr(device_name)}.')
	if devices_info_df.loc[device_name,'Manufacturer'] == 'FBK':
		return f"{devices_info_df.loc[device_name,'Manufacturer']}-W{devices_info_df.loc[device_name,'Wafer']}-{devices_info_df.loc[device_name,'Type']}-{devices_info_df.loc[device_name,'Guard ring']}"
	else:
		raise NotImplementedError(f'Not implemented yet for device manufactured by {devices_info_df.loc[device_name,"Manufacturer"]}.')
	
if __name__ == '__main__':
	for device in devices_info_df.index:
		try:
			print(device, get_device_alias(device))
		except NotImplementedError:
			pass
