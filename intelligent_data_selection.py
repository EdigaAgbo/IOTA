import csv
import pprint
from iota import Iota
from iota import Transaction
from iota import TryteString
import io
import pandas as pd
import numpy as np
from imputation import KNN_imputation,mean_imputation,mode_imputation,reg_imputation,impute_em
# from imputation import rf_imputation
import array
from imputation import rmse, nrmse, r_squared, MAE
import time
import os,psutil
import resource
# README
# To run this code, copy the address from the 'send_multiple-data code
# (change address and copy again to resend new transaction)
# this code takes a large transaction that is split into multiple
# trytes, reattaches and decodes the trytes to load the data sent over the tangle

start_time = time.time()
# Fetch data from the tangle
api = Iota('https://nodes.devnet.iota.org:443')
address = 'KBAPEHDBEE9IKQTPBMWXJHWTIFVOIZFINQRXYZDCDABAACDQPBKZCVVLYBQBPYDCJUQLFPQLMXRMMS999'
transactions = api.find_transactions(addresses=[address,])

# Transactions on this address have been found, next, we pull all hashes
# iteratively to send it to the get_tryte function

hashes = []
for txhash in transactions['hashes']:
    hashes.append(txhash)

# get trytes to accept multiple transaction hashes
trytes = api.get_trytes(hashes)['trytes']
# next, we have to get the trytes from all messages and put them in the right order
# This is done by looking at the index of the transaction
parts = []
for trytestring in trytes:
    tx = Transaction.from_tryte_string(trytestring)
    parts.append((tx.current_index,tx.signature_message_fragment))
parts.sort(key=lambda x: x[0])

# finally, we concatenate and wrap the parts in a Trytestring object
# We further decode the trytestring into python format and
# place our data in a dataframe.
full_message = TryteString.from_unicode('')
for index,part in parts:
    full_message += part
decode = full_message.decode()
data_string = io.StringIO(decode)
tangle_data = pd.read_csv(data_string, sep=",")
full_data = tangle_data.to_numpy()

# tangle_data.to_csv('tangle_data.txt', index=False, header=None, sep =' ', mode='a')
# After downloading the complete data from the tangle, next, we read in our data with missing
# values to perform interpolation or matrix completion
missing_dataset = pd.read_csv('Data_2.csv')
arr = missing_dataset.iloc[:,:2].values
array_len = (len(arr))
# Next step is to try to update missing values from a certain range from the tangle data
# e.g updating missing values using time stamp
print(missing_dataset)
num_missing_data = missing_dataset.isnull().sum(axis = 0).sum()
missing_data_percent = (num_missing_data/array_len)*100
print('Missing values detected at %3f'% missing_data_percent,"%",'in your dataset')
missing_dataset['date'] = pd.date_range('01/02/2021', periods=9971, freq='D')
date_data = (missing_dataset['date'] > '01/01/2021') & (missing_dataset['date'] <= '16/10/2022')
time_range = (missing_dataset.loc[date_data])
# Percentage of missing data in distribution
arr1 = (len(time_range))
time_missing_data = time_range.isnull().sum(axis = 0).sum()
time_missing_data_percent = (time_missing_data/arr1)*100


# This is where the interpolated values with reflect
imputed_data_values = pd.DataFrame(time_range.iloc[:,[1,2]].values)
tangle_values = pd.DataFrame(tangle_data.iloc[:,[1,2]].values)
# These are the tangle values that are interpolated into the "imputed_data_values" dataframe
result = imputed_data_values[imputed_data_values.isnull()] = tangle_values
print(time_range)
print('This distribution contains %3f'% time_missing_data_percent,"%",'missing values \n .... Using time stamp to request data from available node')
print(imputed_data_values)
print('Missing values have been imputed')

# print (imputed_data_values.isnull().sum(axis = 0).sum())




