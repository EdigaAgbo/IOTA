import io
from iota import Iota
from iota import ProposedTransaction
from iota import Address
from iota import Tag
from iota import TryteString
from pprint import pprint
import time
import resource
import pandas as pd
from io import StringIO

start_time = time.time()
api = Iota('https://nodes.devnet.iota.org:443', testnet=True)

TargetAddress1 = 'MBAPEHDBEE9IKQTPBBMWJHWTIFVOIZFINQRXYZDCDABAACDQPBKZCVVLYBQBPYKRUUQLFPQLMXRMMS999'
# Upload data with different percentage of missing data
dataframe = open('Data_1.csv').read()
IO_data = io.StringIO(dataframe)
data = pd.read_csv(IO_data, sep=",")
data['date'] = pd.date_range('01/02/2021', periods=9971, freq='D')
date_data = (data['date'] > '01/01/2021') & (data['date'] <= '16/10/2024')
data_timestamp = (data.loc[date_data])
tx_data = data_timestamp.to_string(index=False,index_names=False).split('\n')
vals = [','.join(ele.split()) for ele in tx_data]
string = '\n'.join([str(item) for item in vals])
message1 = TryteString.from_unicode(string)

pt1 = ProposedTransaction(
address = Address(TargetAddress1),
message = message1,
tag = Tag('SENSORDATA'),
value = 0)

# pt1 = api.prepare_transfer(transfers=[pt1])
result = api.send_transfer(transfers = [pt1])
print(result['bundle'].tail_transaction.hash)
pprint('https://utils.iota.org/transaction/%s/devnet' % result['bundle'][0].hash)
print ("Execution time: {:.2f}s".format(time.time() - start_time))
