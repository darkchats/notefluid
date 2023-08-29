import pandas as pd
from tqdm import tqdm
import numpy as np


def cul_msd(df, col_time="t", col_x="x", col_y="y"):
    df_fill = pd.DataFrame([[i + 1] for i in range(1, int(df[col_time].max()))])
    msd_result = []
    df_fill.columns = [col_time]
    df_fill = pd.merge(df_fill, df, on=col_time, how="left")
    for i in tqdm(df_fill[col_time].drop_duplicates().values):
        df_fill["x1"] = df_fill[col_x].diff(i)
        df_fill["y1"] = df_fill[col_y].diff(i)
        df_fill["T"] = df_fill["x1"] ** 2 + df_fill["y1"] ** 2
        msd_result.append([i, df_fill["T"].sum(), df_fill["T"].count()])
    msd_df = pd.DataFrame(msd_result)
    msd_df.columns = [col_time, "sum", "cnt"]
    msd_df["msd"] = msd_df["sum"] / msd_df["cnt"]
    msd_df = msd_df[msd_df["msd"] > 0]
    msd_df = msd_df.reset_index(drop=True)
    msd_df["v"] = np.sqrt(msd_df["msd"]) / msd_df[col_time]
    msd_df = msd_df[msd_df["step"] > 0]
    return msd_df
