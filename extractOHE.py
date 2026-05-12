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

le = skp.LabelEncoder()
output = le.fit_transform(df['Label'])
print(list(le.classes_))
print(output)

df['Dst Port'] = df['Dst Port'].astype(str)
df[Protocol] = df[Protocol].astype(str)

features_ohe = df[['Dst Port', 'Protocol']]
features_ohe = features_ohe.fillna(0).astype(int).astype(str)
ohe = skp.OneHotEncoder(sparse_output=True, handle_unknown='ignore')
output = ohe.fit_transform(features_ohe)


