import pandas as pd


def load_v(path):
    with open(path, 'r') as fr:
        row1 = fr.readline()
        row2 = fr.readline()
    parameter_map = dict(kv.strip().split('=') for kv in row2.strip("zone").split(','))
    df = pd.read_csv(path, skiprows=2, nrows=int(parameter_map['I']) * int(parameter_map['J']) - 1, sep='\s+')
    cols = row1.strip('ZIBE').split('=')[1].strip().replace('"', '').split(',')
    cols = [col for col in cols if len(col) > 0]
    df.columns = cols
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
