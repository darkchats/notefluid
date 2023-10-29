import pandas as pd


def load_v(path):
    with open(path, 'r') as fr:
        row1 = fr.readline()
        row2 = fr.readline()
    parameter_map = dict(kv.strip().split('=') for kv in row2.strip("zone").split(','))
    df = pd.read_csv(path, skiprows=2, nrows=int(parameter_map['I']) * int(parameter_map['J']) - 1, sep='\s+')
    df.columns = row1.strip('ZIBE').split('=')[1].strip().replace('"', '').split(',')
    df[['x', 'y']] = df[['x', 'y']].astype('int')
    return df


def load_p(path):
    with open(path, 'r') as fr:
        row1 = fr.readline()
        row2 = fr.readline()
    parameter_map = dict(kv.strip().split('=') for kv in row2.split(','))
    df = pd.read_csv(path, skiprows=2, nrows=int(parameter_map['E']), sep='\s+')
    df.columns = row1.split('=')[1].strip().replace('"', '').split(',')
    return df