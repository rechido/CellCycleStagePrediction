import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalize(df: pd.DataFrame, normalization_type) -> pd.DataFrame:
 
    if normalization_type == 'whitening':
        std = df.std()
        if type(std) == float:
            if std == 0:
                std = 0.00001
        else:
            std[std == 0] = 0.00001
        normalized_df = (df - df.mean()) / std

    elif normalization_type == 'min_max_scaling':
        diff = df.max() - df.min()
        if type(diff) == float:
            if diff == 0:
                diff = 0.00001
        else:
            diff[diff == 0] = 0.00001
        normalized_df = (df - df.min()) / diff

    elif normalization_type == 'log':
        normalized_df = np.log(df + 1)

    else:
        assert(False, 'invalid normalization value')

    return normalized_df


