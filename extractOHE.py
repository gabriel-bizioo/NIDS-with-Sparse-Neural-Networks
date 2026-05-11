import argparse
import sys
import pandas as pd
from sklearn import preprocessing as skp

parser = argparse.ArgumentParser(sys.argv[0])
parser.add_argument("file", help="File to be loaded and processed", type=str)
args = parser.parse_args()

columns = ['Dst Port', 'Protocol', 'Label']
df = pd.read_csv(args.file, usecols=columns, low_memory=False)[columns]
print(df.head())

labels = ['BENIGN', 'Botnet', 'Botnet - Attempted', 'Portscan', 'DDoS']
le = skp.LabelEncoder()
le.fit(labels)
list(le.classes_)
