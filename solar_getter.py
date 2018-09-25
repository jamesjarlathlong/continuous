#import additional module for SAM simulation:
from py3samsdk import PySSC
import pandas as pd
import pandas as pd
import numpy as np
import sys
import os
import json
def get_solar(fname, year, lat, lon, leap_year, interval,
			  utc, your_name, your_email,
			  your_affiliation, api_key,attributes):
	urls = ('http://developer.nrel.gov/api/solar/nsrdb_0512_download.csv?'
		 'wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}'
		 '&interval={interval}&utc={utc}&full_name={name}&email={email}'
		 '&affiliation={affiliation}'
		 '&api_key={api}&attributes={attr}')
	fmatted_url = urls.format(year=year, lat=lat, lon=lon,
					      leap=leap_year, interval=interval,
					      utc=utc, name=your_name, email=your_email,
					      affiliation=your_affiliation, api=api_key, attr=attributes)
	info = pd.read_csv(fmatted_url, nrows=1, dtype={"Time Zone":int, "Elevation":int})
	# See metadata for specified properties, e.g., timezone and elevation
	metadata = info.iloc[0].to_dict()
	print('metadata: ', metadata)
	df = pd.read_csv(fmatted_url,skiprows=2)
	df = df.set_index(pd.date_range('1/1/{yr}'.format(yr=year), freq=interval+'Min', periods=525600/int(interval)))
	df.to_pickle(fname+'.pkl')
	with open(fname+'.metadata.json', 'w') as f:
		json.dump({k:str(v) for k,v in metadata.items()}, f)
	return metadata, df

def convert_to_energy(cell_properties, metadata, df):
	ssc_lib = os.environ.get('LD_LIBRARY_PATH')
	ssc = PySSC(ssc_lib)
	# Resource inputs for SAM model:
	wfd = ssc.data_create()
	ssc.data_set_number(wfd, 'lat', metadata['Latitude'])
	ssc.data_set_number(wfd, 'lon', metadata['Longitude'])
	ssc.data_set_number(wfd, 'tz', metadata['Time Zone'])
	ssc.data_set_number(wfd, 'elev', metadata['Elevation'])
	ssc.data_set_array(wfd, 'year', df.index.year)
	ssc.data_set_array(wfd, 'month', df.index.month)
	ssc.data_set_array(wfd, 'day', df.index.day)
	ssc.data_set_array(wfd, 'hour', df.index.hour)
	ssc.data_set_array(wfd, 'minute', df.index.minute)
	ssc.data_set_array(wfd, 'dn', df['DNI'])
	ssc.data_set_array(wfd, 'df', df['DHI'])
	ssc.data_set_array(wfd, 'wspd', df['Wind Speed'])
	ssc.data_set_array(wfd, 'tdry', df['Temperature'])
	# Create SAM compliant object  
	dat = ssc.data_create()
	ssc.data_set_table(dat, 'solar_resource_data', wfd)
	ssc.data_free(wfd)
	# Specify the system Configuration
	# Set system capacity in kW
	ssc.data_set_number(dat, 'system_capacity', cell_properties['system_capacity'])
	# Set DC/AC ratio (or power ratio). See https://sam.nrel.gov/sites/default/files/content/virtual_conf_july_2013/07-sam-virtual-conference-2013-woodcock.pdf
	ssc.data_set_number(dat, 'dc_ac_ratio',1.0)
	# Set tilt of system in degrees
	ssc.data_set_number(dat, 'tilt', cell_properties['tilt'])
	# Set azimuth angle (in degrees) from north (0 degrees)
	ssc.data_set_number(dat, 'azimuth', cell_properties['azimuth'])#i.e. south
	# Set the inverter efficency
	ssc.data_set_number(dat, 'inv_eff', 99)
	# Set the system losses, in percent
	ssc.data_set_number(dat, 'losses', 14.0757)
	# Specify fixed tilt system (0=Fixed, 1=Fixed Roof, 2=1 Axis Tracker, 3=Backtracted, 4=2 Axis Tracker)
	ssc.data_set_number(dat, 'array_type', 0)
	# Set ground coverage ratio
	ssc.data_set_number(dat, 'gcr', 1.0)
	# Set constant loss adjustment
	ssc.data_set_number(dat, 'adjust:constant', 0)
	# execute and put generation results back into dataframe
	mod = ssc.module_create('pvwattsv5')
	ssc.module_exec(mod, dat)
	generation = np.array(ssc.data_get_array(dat, 'gen'))
	dcnet = np.array(ssc.data_get_array(dat, 'dc'))
	ac = np.array(ssc.data_get_array(dat, 'ac'))
	# free the memory
	ssc.data_free(dat)
	ssc.module_free(mod)
	return generation, dcnet, ac

if __name__ == '__main__':
	api_key = os.environ.get('NREL')
	#lat, lon= 42.3583452, -71.0937524
	lat, lon= 42.3683452, -71.0957524
	# Set the attributes to extract (e.g., dhi, ghi, etc.), separated by commas.
	attributes = 'ghi,dhi,dni,wind_speed_10m_nwp,surface_air_temperature_nwp,solar_zenith_angle'
	# Choose year of data
	year = '2011'
	# Set leap year to true or false. True will return leap day data if present, false will not.
	leap_year = 'false'
	# Set time interval in minutes, i.e., '30' is half hour intervals. Valid intervals are 30 & 60.
	interval = '30'
	# Specify Coordinated Universal Time (UTC), 'true' will use UTC, 'false' will use the local time zone of the data.
	# NOTE: In order to use the NSRDB data in SAM, you must specify UTC as 'false'. SAM requires the data to be in the
	# local time zone.
	utc = 'false'
	# Your full name, use '+' instead of spaces.
	your_name = 'James+Long'
	# Your reason for using the NSRDB.
	# Your email address
	your_email = 'jjlong@mit.edu'
	# Please join our mailing list so we can keep you up-to-date on new developments.
	mailing_list = 'true'
	reason_for_use = 'beta+testing'
	# Your affiliation
	your_affiliation = 'mit'
	# Your email address
	fname = 'killian_court'
	meta, df = get_solar(fname, year=year, lat=lat, lon=lon,
			leap_year=leap_year, interval=interval,
			utc=utc, your_name=your_name, your_email=your_email,
			your_affiliation=your_affiliation,
			api_key=api_key, attributes=attributes)
	cell_properties = {'system_capacity':2e-3 , 'azimuth':180 , 'tilt':0}
	gen = convert_to_energy(cell_properties, meta, df)
	print(gen)
	"""
	df = pd.read_csv(fmatted_url,skiprows=1)
	df = df.set_index(pd.date_range('1/1/{yr}'.format(yr=year), freq=interval+'Min', periods=525600/int(interval)))
	ssc_lib = '/home/jjlong/SAMSDK/linux64/'
	ssc = PySSC(ssc_lib)
	# Resource inputs for SAM model:
	wfd = ssc.data_create()
	ssc.data_set_number(wfd, 'lat', lat)
	ssc.data_set_number(wfd, 'lon', lon)
	ssc.data_set_number(wfd, 'tz', timezone)
	ssc.data_set_number(wfd, 'elev', elevation)
	ssc.data_set_array(wfd, 'year', df.index.year)
	ssc.data_set_array(wfd, 'month', df.index.month)
	ssc.data_set_array(wfd, 'day', df.index.day)
	ssc.data_set_array(wfd, 'hour', df.index.hour)
	ssc.data_set_array(wfd, 'minute', df.index.minute)
	ssc.data_set_array(wfd, 'dn', df['DNI'])
	ssc.data_set_array(wfd, 'df', df['DHI'])
	ssc.data_set_array(wfd, 'wspd', df['Wind Speed'])
	ssc.data_set_array(wfd, 'tdry', df['Temperature'])

	# Create SAM compliant object  
	dat = ssc.data_create()
	ssc.data_set_table(dat, 'solar_resource_data', wfd)
	ssc.data_free(wfd)

	# Specify the system Configuration
	# Set system capacity in MW
	system_capacity = 4
	ssc.data_set_number(dat, 'system_capacity', system_capacity)
	# Set DC/AC ratio (or power ratio). See https://sam.nrel.gov/sites/default/files/content/virtual_conf_july_2013/07-sam-virtual-conference-2013-woodcock.pdf
	ssc.data_set_number(dat, 'dc_ac_ratio', 1.1)
	# Set tilt of system in degrees
	ssc.data_set_number(dat, 'tilt', 25)
	# Set azimuth angle (in degrees) from north (0 degrees)
	ssc.data_set_number(dat, 'azimuth', 180)
	# Set the inverter efficency
	ssc.data_set_number(dat, 'inv_eff', 96)
	# Set the system losses, in percent
	ssc.data_set_number(dat, 'losses', 14.0757)
	# Specify fixed tilt system (0=Fixed, 1=Fixed Roof, 2=1 Axis Tracker, 3=Backtracted, 4=2 Axis Tracker)
	ssc.data_set_number(dat, 'array_type', 0)
	# Set ground coverage ratio
	ssc.data_set_number(dat, 'gcr', 0.4)
	# Set constant loss adjustment
	ssc.data_set_number(dat, 'adjust:constant', 0)

	# execute and put generation results back into dataframe
	mod = ssc.module_create('pvwattsv5')
	ssc.module_exec(mod, dat)
	df['generation'] = np.array(ssc.data_get_array(dat, 'gen'))

	# free the memory
	ssc.data_free(dat)
	ssc.module_free(mod)
	"""