#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from classifier import TransactionClassifier
import pandas
import sys

if(len(sys.argv) < 6):
	print("Invalid parameters. Usage: example.py [dataset_to_train_path] [dataset_to_predict_path] [transaction_column] [category_column] [dataset_to_train_test_percentage] [bow_max_features] [features_percentage]")
else: 
	train_dataset = pandas.read_csv(sys.argv[1])
	predict_dataset = pandas.read_csv(sys.argv[2])

	transaction_column = int(sys.argv[3])
	category_column = int(sys.argv[4])
	dataset_train_test_percentage = int(sys.argv[5])
	bow_max_features = int(sys.argv[6])
	features_percentile = int(sys.argv[7])

	classifier = TransactionClassifier(train_dataset, transaction_column, category_column, dataset_train_test_percentage, bow_max_features, features_percentile)
	predictions = classifier.predict_dataset(predict_dataset, transaction_column)
	print(predictions)

