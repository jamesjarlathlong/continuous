import solar_getter


def battery_dynamics(maxbatt, status, battery):
    if status == 2:#sleeping
        new_battery = min(battery+1, maxbatt)
    else:#either pre-sleep or awake
        new_battery = max(0, battery-1)
    return new_battery

def battery_dynamics(generated_power, maxbatt, status, battery):
    discharge_voltage = 3.7 #Volts
    timeperiod = 0.5 #hours
    added_power =  generated_power*1000*timeperiod #mWh - generated is avg power in a timeperiod
    max_possible = maxbatt*discharge_voltage #mAh e.g. 2000mAh times 3.7 volts = 7400mW 
    on_power = 56+45+5#mAh
    off_power = 0.5#mAh
    if status == 2:#sleeping
        balance = added_power - off_power
        new_battery = min(battery+balance, maxbatt)
    else:#either pre-sleep or awake
        balance = added_power - on_power
        new_battery = max(0, battery+balance)
    return new_battery

def slice_solar(solardf, start, duration):
	"""solar_getter's get_solar function returns 
	metadata and a dataframe, return a slice of the dataframe
	beginning with start and lasting duration timeslice"""
	pass
#Given the full solar meteorological time series
#for the given run, product a generated watts time series
