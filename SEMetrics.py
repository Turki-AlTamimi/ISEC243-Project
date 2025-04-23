# --------------------------------------------------------------------------------------------------------------
#
# Filename: SEMetrics.py
# Author: 
# Dated: 
# SEMetrics => Software Engineering Metrics for Sifting Android Malicious Applications
# For more details read the project report
#
# --------------------------------------------------------------------------------------------------------------

from __future__ import print_function
import os, sys, argparse
import pandas as pd

from Classify import *

__DEBUG__  = True
CLASS_LABEL = "Class"
__WITH_APK__ = False
# Classes
classes = ['benign', 'malware']

if __name__=="__main__":
	print("--- SEMetrics START ---")

	parser = argparse.ArgumentParser()

	parser.add_argument("-f", type=str)
	parser.add_argument("-csv", type=str)
	args = parser.parse_args()

	if args.csv == None:
		print("ERROR:SEMetrics: Missing arguments\nUsage:\n")
		print("   python SEMetrics.py -csv <csv_filename>")
		sys.exit(1)
	csv_filename = args.csv.strip()
	if csv_filename is None:
		print("Please enter the filename of the csv file")
		print("   python SEMetrics.py -csv <csv_filename>")
		sys.exit(1)

	# If the features have not already been extracted,
	# extract the features from the generated CSV files.
	# It assumes that the method level features for APKs
	# have already been collected in respective APK CSV
	# files. It then calculates different metrics for each
	# APK and combines them all in one CSV file for clasification.
	if args.f == None:
		print("ERROR:SEMetrics: Missing arguments\nUsage:\n")
		print("   python SEMetrics.py -f <file_containing_csv_files> -csv <csv_filename>")
		sys.exit(1)
	file_containing_csv_files = args.f.strip()
	if file_containing_csv_files is None:
		print("Please enter the filename containing the list of CSV files")
		print("   python SEMetrics.py -f <file_containing_csv_files> -csv <csv_filename>")
		sys.exit(1)
	if not os.path.isfile(file_containing_csv_files):
		print("File '%s' does not exist"%file_containing_csv_files)
		print("   python SEMetrics.py -f <file_containing_csv_files> -csv <csv_filename>")
		sys.exit(1)

	file_containing_csv_files = os.path.abspath(file_containing_csv_files)
	print("Reading list of files from %s"%file_containing_csv_files)
	f = open(file_containing_csv_files, 'r')
	files = f.readlines()
	f.close()

    #
    # IMPEMENT THE REST
    #

	print("--- SEMetrics END ---")
