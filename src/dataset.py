import pandas as pd
import sklearn.preprocessing as pre
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch
import numpy as np


class DataTransformer:
    def __init__(self, encoders=None, scaler=None, is_train_mode=True):
        self.cont_cols = [
            "startCount",
            "clickCount",
            "installCount",
            "startCount1d",
            "startCount7d",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "elapsed_time",
            "countRatio",
        ]
        self.cat_cols = [
            "campaignId",
            "platform",
            "softwareVersion",
            "sourceGameId",
            "country",
            "connectionType",
            "deviceType",
        ]
        self.is_train_mode = is_train_mode
        self.encoders = encoders
        self.scaler = scaler

    def _clean_data(self, data: pd.DataFrame):
        df = data.copy()
        df.loc[df.country.isna(), "country"] = "UNK"  # UNK for 'Unknown'
        df.loc[df.lastStart.isna(), "lastStart"] = df.timestamp[df.lastStart.isna()]

        # deviceType missing values
        tmp = df.loc[
            (
                (df.platform == "android")
                & (df.softwareVersion == "4.4.2")
                & (df.country.isin(["KR", "SE"]))
            )
        ].copy()

        # Fill na with the mode
        df.loc[:, "deviceType"].fillna(tmp.deviceType.mode()[0], inplace=True)
        return df

    def _preprocess_cat_data(self, data: pd.DataFrame):
        df = data.copy()
        if self.is_train_mode:
            self.encoders = {
                col: pre.OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=len(df.loc[:, col].unique()) + 1
                ).fit(np.array(df.loc[:, col].values).reshape(-1, 1))
                for col in self.cat_cols
            }

        for col in self.cat_cols:
            df.loc[:, col] = self.encoders[col].transform(np.array(df.loc[:, col].values).reshape(-1, 1))
        df.loc[:, self.cat_cols] = df.loc[:, self.cat_cols].astype("int32")

        return df

    def _preprocess_cont_data(self, data):
        df = data.copy()
        df.loc[:, "timestamp"] = pd.to_datetime(df.timestamp)
        df.loc[:, "month"] = df.timestamp.dt.month
        df.loc[:, "day"] = df.timestamp.dt.day
        df.loc[:, "hour"] = df.timestamp.dt.hour
        df.loc[:, "minute"] = df.timestamp.dt.minute
        df.loc[:, "second"] = df.timestamp.dt.second

        df.loc[:, "lastStart"] = pd.to_datetime(df.lastStart)
        df.loc[:, "elapsed_time"] = (
            df.timestamp - df.lastStart
        ).dt.total_seconds() / 3600.0

        df.loc[:, "countRatio"] = df.viewCount / df.startCount

        df.drop(["timestamp", "lastStart", "viewCount"], axis=1, inplace=True)

        if self.is_train_mode:
            self.scaler = StandardScaler()
            self.scaler.fit(df.loc[:, self.cont_cols].values)
        scaled = self.scaler.transform(df.loc[:, self.cont_cols].values)
        df.loc[:, self.cont_cols] = scaled

        df.loc[:, self.cont_cols] = df.loc[:, self.cont_cols].astype("float64")

        return df

    def transform(self, data):
        df = self._clean_data(data)
        df = self._preprocess_cat_data(df)
        df = self._preprocess_cont_data(df)
        return df


class CustomDataset(Dataset):
    def __init__(self, emb_cols, cont_cols, x, y):
        cat_data = x.loc[:, emb_cols]
        self.cat_data = np.stack(
            [c.values for n, c in cat_data.items()], axis=1
        ).astype(np.int64)
        cont_data = x.loc[:, cont_cols]
        self.cont_data = np.stack(
            [c.values for n, c in cont_data.items()], axis=1
        ).astype(np.float32)
        self.y = y.values.astype(np.int32) if y is not None else np.repeat(0, len(x))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        cont_data = self.cont_data[item]
        to_embed = self.cat_data[item]
        y = np.asarray(self.y[item])

        out = {
            "x_cont": torch.from_numpy(cont_data),
            "x_emb": torch.from_numpy(to_embed),
            "y": torch.from_numpy(y),
        }

        return out
