import argparse
from ctypes import sizeof
import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing as skp
from sklearn.datasets import  dump_svmlight_file

parser = argparse.ArgumentParser(sys.argv[0])
parser.add_argument("file", help="File to be loaded and processed", type=str)
args = parser.parse_args()

columns = ['Dst Port', 'Protocol', 'Packet Length Std', 'Label']
df = pd.read_csv(args.file, usecols=columns, low_memory=False)[columns]
print(df.head())

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

le = skp.LabelEncoder()
labels = le.fit_transform(df['Label'])
print(list(le.classes_))
count = list(le.classes_)
print(count.__sizeof__())
print(labels.size)

ports = [str(i) for i in range(65536)]
protocols = [str(i) for i in range(256)]

bins_pkt = [0, 64, 512, 1000, 1500, float('inf')]
labels_pkt = ['Tiny', 'Small', 'Medium', 'Large', 'Jumbo']
df['Pkt_Bin'] = pd.cut(df['Packet Length Std'], bins=bins_pkt, labels=labels_pkt)

ohe = skp.OneHotEncoder(categories=[ports, protocols], sparse_output=True, handle_unknown='ignore')
features_ohe = df[['Dst Port', 'Protocol', 'Pkt_Bin']]
features_ohe = features_ohe.fillna(0).astype(int).astype(str)


output = ohe.fit_transform(features_ohe)
print(output.shape)
print(ohe.get_feature_names_out(['Dst Port', 'Protocol']))

#dump_svmlight_file(output, labels, 'SparseCICIDSTest.txt', zero_based=True)
