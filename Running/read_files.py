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

activities_loc = "D:/maps/Running/Data/kg_06142024/activities/"

def get_file_names(directory):

    names = []
    types = []
    zipped = []
    files = []
    paths = []
    activity_directory = os.fsencode(directory)

    for i in os.listdir(directory):
    	paths.append(directory + i)
    	split_i = i.split(".")
    	names.append(split_i[0])
    	types.append(split_i[1])
    	if len(split_i) == 3:
    		zipped.append(1)
    	else:
    		zipped.append(0)

    return names, types, zipped, paths


def read_gpx_files(files, types, paths):

	bad_files = []
	activity = []
	activity_type = []
	name = []
	latitude = []
	longitude = []
	elevation = []
	timestamp = []

	for i in range(0, len(files)):
		
		print("Reading file", i + 1, "of", len(files))
		if types[i] == "gpx":
			try:
				gpx_file = open(paths[i], "r", encoding = "utf8")
				gpx_data = gpxpy.parse(gpx_file)
			except:
				bad_files.append(files[i])
				pass

			for track in gpx_data.tracks:
				for segment in track.segments:
					for point in segment.points:
						activity.append("gpx_" + str(i))
						activity_type.append(track.type)
						name.append(track.name)
						latitude.append(point.latitude)
						longitude.append(point.longitude)
						elevation.append(point.elevation)
						timestamp.append(point.time)

	print(len(set(activity)), "files read.", len(bad_files),
		"files are bad.", len(files)-(len(set(activity))+len(bad_files)),
		"are not gpx files.")

	return bad_files, activity, activity_type, name, latitude, longitude, elevation, timestamp


def read_zipped_fit_files(files, types, paths, zipped):

	# bad_files = []
	activity = []
	activity_type = []
	name = []
	latitude = []
	longitude = []
	elevation = []
	timestamp = []

	for i in range(0, len(files)):

		print("Reading file", i + 1, "of", len(files))

		if types[i] == "fit" and zipped[i] == 1:

			fit_file = gzip.open(paths[i])
			fit_file_read = fit_file.read()
			fit_file.close()

			fit_file = FitFile(fit_file_read)

			for session in fit_file.get_messages("session"):
				session_dict = session.get_values()
				activity_type_i = session_dict["sport"]
				# print(session_dict["event"])
				# print(session_dict["event_type"])
				# print(session_dict["sub_sport"])

			# missing = 0
			for record in fit_file.get_messages("record"):
				record_dict = record.get_values()

				activity.append(i)
				activity_type.append(activity_type_i)
				name.append(files[i])
				try:
					timestamp.append(record_dict["timestamp"])
				except KeyError:
					timestamp.append(None)
				try:
					latitude.append(record_dict["position_lat"] / ((2**32)/360))
				except (KeyError, TypeError):
					latitude.append(None)
				try:
					longitude.append(record_dict["position_long"] / ((2**32)/360))
				except (KeyError, TypeError):
					longitude.append(None)
				try:
					elevation.append(record_dict["enhanced_altitude"])
				except KeyError:
					elevation.append(None)					
				# for data in record:
				# 	if data.name == "timestamp":
				# 		if data.value is not None:
				# 			timestamp.append(data.value)
				# 		else:
				# 			timestamp.append(None)
				# 	if data.name == "position_lat":
				# 		activity.append(i)
				# 		if data.value is not None:
				# 			latitude.append(data.value / ((2**32)/360))
				# 		else:
				# 			missing += 1
				# 			latitude.append(None)
				# 	if data.name == "position_long":
				# 		if data.value is not None:
				# 			longitude.append(data.value / ((2**32)/360))
				# 		else:
				# 			longitude.append(None)
				# 	if data.name == "enhanced_altitude":
				# 		if data.value is not None:
				# 			elevation.append(data.value)
				# 		else:
				# 			elevation.append(None)

	print(len(set(activity)), "files read.", len(files)-len(set(activity)),
		"are not zipped fit files.")

	return activity, activity_type, name, latitude, longitude, elevation, timestamp

strava_files, strava_types, strava_gz, strava_paths = get_file_names(directory = activities_loc)

gpx_bad_files, gpx_activity, gpx_activity_type, gpx_name, gpx_latitude, gpx_longitude, gpx_elevation, gpx_timestamp = read_gpx_files(
	files = strava_files,
	types = strava_types,
	paths = strava_paths
	)

gpx_df = pd.DataFrame(
	list(zip(gpx_timestamp, gpx_activity, gpx_activity_type, gpx_name, gpx_latitude, gpx_longitude, gpx_elevation)),
	columns = ["Time", "Activity", "Type", "Name", "Latitude", "Longitude", "Elevation"]
	)

gpx_df.to_pickle("D:/maps/Running/Data/kg_06142024_gpx.pkl")

fit_activity, fit_activity_type, fit_name, fit_latitude, fit_longitude, fit_elevation, fit_timestamp = read_zipped_fit_files(
	files = strava_files,
	types = strava_types,
	paths = strava_paths,
	zipped = strava_gz
	)

fit_df = pd.DataFrame(
	list(zip(fit_timestamp, fit_activity, fit_activity_type, fit_name, fit_latitude, fit_longitude, fit_elevation)),
	columns = ["Time", "Activity", "Type", "Name", "Latitude", "Longitude", "Elevation"]
	)

fit_df.to_pickle("D:/maps/Running/Data/kg_06142024_fit.pkl")
