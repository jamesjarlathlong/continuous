import solar_getter
import pandas as pd
import json
import os
import itertools
import functools
import numpy as np
import sys
if __name__ == '__main__':
	api_key = os.environ.get('NREL')
	lat, lon= 42.3683452, -71.0957524
	# Set the attributes to extract (e.g., dhi, ghi, etc.), separated by commas.
	attributes = 'ghi,dhi,dni,wind_speed_10m_nwp,surface_air_temperature_nwp,solar_zenith_angle'
	# Choose year of data
	
	# Set leap year to true or false. True will return leap day data if present, false will not.
	leap_year = 'false'
	# Set time interval in minutes, i.e., '30' is half hour intervals. Valid intervals are 30 & 60.	interval = '30'
	# Specify Coordinated Universal Time (UTC), 'true' will use UTC, 'false' will use the local time zone of the data.
	# NOTE: In order to use the NSRDB data in SAM, you must specify UTC as 'false'. SAM requires the data to be in the
	# local time zone.
	utc = 'false'
	# Your full name, use '+' instead of spaces.
	your_name = 'James+Long'
	# Your reason for using the NSRDB.
	#	 Your email address
	your_email = 'jjlong@mit.edu'
	mailing_list = 'true'
	reason_for_use = 'beta+testing'
	your_affiliation = 'mit'
	interval='30'
	year = '2012'
	fname='training_12'
	meta, df = solar_getter.get_solar(fname, year=year, lat=lat, lon=lon,
                    leap_year=leap_year, interval=interval,
                    utc=utc, your_name=your_name, your_email=your_email,
                    your_affiliation=your_affiliation,
                    api_key=api_key, attributes=attributes)