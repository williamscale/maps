import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import gzip
import collections.abc
collections.Iterable = collections.abc.Iterable

from fitparse import FitFile
import gpxpy
import gpxpy.gpx

# import geopandas as gpd
# from shapely.geometry import Point, Polygon

# activities_loc = "C:/Users/caler/Documents/MyProjects/Running/strava_081523/activities/"
activities_loc = "C:/Users/caler/Documents/MyProjects/Running/strava_kg_082123/activities/"
def get_file_names(directory):

    names = []
    types = []
    zipped = []
    files = []
    activity_directory = os.fsencode(directory)
    # print(activity_directory)

    for i in os.listdir(directory):
    	# print(i)
    	split_i = i.split(".")
    	names.append(split_i[0])
    	types.append(split_i[1])
    	if len(split_i) == 3:
    		zipped.append(1)
    	else:
    		zipped.append(0)
    	# files.append(i.split("."))
    	# files.append(i)
    	# split_tup = os.path.splitext(i)
    	# split_tup = os.path.split(os.extsep, 1)
    	# files.append(split_tup)


    # for file in os.listdir(activity_directory):
    #     file_dec = os.fsdecode(file)
    #     if file_dec.endswith(suffix):
    #        file_name_list.append(file)
    # else:
    #     pass

    return names, types, zipped

strava_files, strava_types, strava_gz = get_file_names(directory = activities_loc)


# print(strava_types)
# gpx_file_name_list = get_file_names(directory)
# print(fit_file_name_list)
# print(gpx_file_name_list)

def get_activity_history(directory, files, file_types, file_zipped):

	activity = []
	name = []
	latitude = []
	longitude = []
	elevation = []

	for i in range(0, len(files)):
		print(i)
	# range(0, 1):
	# range(0, len(files)):
		
		if file_zipped[i] == 0:

			loc = directory + files[i] + "." + file_types[i]
			# print(loc)

			if file_types[i] == "fit":
				print("unzipped fit")
				
			elif file_types[i] == "gpx":
				print(loc)
				# gpx_file = open(loc, "r")
				gpx_file = open(loc, "r", encoding="utf8")
				gpx = gpxpy.parse(gpx_file)
				# print(gpx)
				for track in gpx.tracks:
					# print(track.name)
					for segment in track.segments:
						for point in segment.points:
							activity.append(i)
							name.append(track.name)
							latitude.append(point.latitude)
							longitude.append(point.longitude)
							elevation.append(point.longitude)
							# print('Point at ({0},{1}) -> {2}'.format(point.latitude, point.longitude, point.elevation))
			else:
				print("no")

		elif (file_zipped[i] == 1) & (file_types[i] == "fit"):
			loc = directory + files[i] + "." + file_types[i] + ".gz"
			print(loc)
			fit_file = gzip.open(loc)
			fit_file_read = fit_file.read()
			fit_file.close()
			# print(file_zipped[i])
			# print(file_types[i])
			# print(files[i])
			fit_file = FitFile(fit_file_read)
			missing = 0
			for record in fit_file.get_messages("record"):
				for data in record:
					activity.append(i)
					if data.name == "position_lat":
						if data.value is not None:
							latitude.append(data.value / ((2**32)/360))
						else:
							missing += 1
							latitude.append(0)
					if data.name == "position_long":
						if data.value is not None:
							longitude.append(data.value / ((2**32)/360))
						else:
							longitude.append(0)
					if data.name == "altitude":
						elevation.append(data.value)
					# if data.units:
					# 	print(" * {}: {} ({})".format(data.name, data.value, data.units))
					# else:
					# 	print(" * {}: {}".format(data.name, data.value))
				# print("---") 
			# fit_file = gzip.open(loc)
			# fit_file_read = fit_file.read()
			# fit_file.close()
			# fit_file_pars = FitFile(fit_file_read)
			# print(fit_file_pars)
		else:
			print("skip")

	print(len(activity))
	print(len(latitude))
	print(len(longitude))
	print(len(elevation))
	df2 = pd.DataFrame(list(zip(latitude, longitude)), columns = ['latitude', 'longitude'])
	df = pd.DataFrame(list(zip(activity, latitude, longitude, elevation)), columns = ['activity', 'latitude', 'longitude', 'elevation'])
	# sa = df[(df.latitude >= 28) & (df.latitude <= 30) & (df.longitude >= -99) & (df.longitude <= -97)]
	# df_406 = df[df.activity == 406]
	print(df2)
	# print(df2['activity'].nunique())


	# plt.scatter(sa.longitude, sa.latitude)
	# plt.scatter(df2.longitude, df2.latitude)
	# plt.show()
	# for i in files:
	# 	print(i[-3:])
		
	# 	if i[-3:] == ".gz":
	# 		print(i, " is zipped.")

	# 	elif i[-3:] == ".gpx":
	# 		print(i)

	# 	else:
	# 		print("not")

	# activity_history = []



    # activity_directory = os.fsencode(dir_as_str)
    
    # for run_id, zipped_file in enumerate(file_name_list):

    # 	zipped_file_path = os.path.join(activity_directory, zipped_file)
    # 	# print(zipped_file_path)

    # 	if file_type == "gpx":

    # 		gpx_file = open(zipped_file_path, 'r')
    # 		gpx = gpxpy.parse(gpx_file)
    # 		# print(gpx.tracks)
    # 		for track in gpx.tracks:
    # 			for segment in track.segments:
    # 				for point in segment.points:
    # 					# print('.')
    # 					print('Point at ({0},{1}) -> {2}'.format(point.latitude, point.longitude, point.elevation))

        # if file_type == "fit"
        # fit_file = gzip.open(zipped_file_path)
        # fit_file_read = fit_file.read()
        # fit_file.close()
        # fit_file_pars = FitFile(fit_file_read)
        # activity = []
        # for record in fit_file_pars.get_messages('record'):
        #     record_dict = {}
        #     for record_content in record:
        #         record_dict[record_content.name] = record_content.value
        #     activity.append(record_dict)
        # activity_history.extend(activity)

        # num_files = len(file_name_list)
        # if run_id % round(num_files * 0.1, 0) == 0:
        #     pct_compl = round(run_id / num_files, 2) * 100
        #     print(f'''{pct_compl}% completed''')

	return df2


df = get_activity_history(activities_loc, files = strava_files, file_types = strava_types, file_zipped = strava_gz)
df.to_csv("C:/Users/caler/Documents/MyProjects/Running/kg_strava_082123.csv")

# df = pd.read_csv('strava_081523.csv', usecols = ["latitude", "longitude"])
# sa = df[(df.latitude >= 29.2) & (df.latitude <= 29.7) & (df.longitude >= -98.8) & (df.longitude <= -98.1)]
# # plt.scatter(sa.longitude, sa.latitude)
# # plt.show()

# crs = {'init':'epsg:4326'}
# geometry = [Point(xy) for xy in zip(sa['longitude'], sa['latitude'])]
# geo_df = gpd.GeoDataFrame(sa, crs = crs, geometry=geometry)
# fig, ax = plt.subplots(figsize=(15,15))
# geo_df.plot(ax = ax)
# activity_history = get_activity_history(dir_as_str, file_type = "gpx", file_name_list = gpx_file_name_list)
# print(activity_history)

# activity_df = pd.DataFrame(activity_history)
# activity_df.set_index('timestamp', drop=True, inplace=True)
# activity_df.to_csv('activity_history_7_july_2020.csv')

# activity_df.loc['2023-05-10', 'heart_rate'].plot()