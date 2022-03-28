import pandas

def read_devices_info_sheet():
	"""Reads the `devices.xlsx` file where all the info about each device
	is supposed to be stored and returns it as a `pandas.dataframe`.
	"""
	return pandas.read_excel('../../doc/devices.xlsx').set_index('device_name')

devices_info_df = read_devices_info_sheet() # This is here such that we can import this object and only read the file once.
