import os
import math
import numpy as np
import pandas as pd
from typing import List

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OrdinalEncoder,
    OneHotEncoder,
    LabelEncoder,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

import random

from utils.utils import type_assertion, read_to_frame


class Analyzer:
    """Analyzer class for data cleaning and exploration processes."""

    # Basic Methods
    def __init__(
        self,
        df: pd.DataFrame,
        target_labels: str | List[str] = [],
        seed: int | float = 0,
        dir_path: str = "",
    ):
        type_assertion(df, pd.DataFrame), type_assertion(seed, int | float)
        self.df: pd.DataFrame = df
        if type(target_labels) is str:
            try:
                assert target_labels in df.columns
            except:
                raise KeyError(
                    f"Target label must be one of the names of the columns in self.df: {self.df.columns}."
                )
            self.target_labels = [target_labels]
        else:
            type_assertion(target_labels, list)
            self.target_labels: List[str] = target_labels
        self.sample_dfs: dict = {}
        self.n_samples: int = self.df.shape[0]
        self.seed: int | float = seed
        self.dir_path = dir_path
        self.n_targets = len(self.target_labels)
        if self.n_targets != 0:
            type_assertion(target_labels, str, iterable=True)
            try:
                for name in target_labels:
                    assert name in self.df.columns
            except:
                raise IndexError(
                    "Column names must all be found in column names of self.df."
                )
        self.n_features = len(self.df.columns) - self.n_targets

    def __str__(self) -> None:
        """Prints basic information about the dataset."""
        if self.target_labels is not None:
            tar = f"\nThe target/label is {self.target_labels}."
        else:
            tar = ""
        ret = (
            f"There are {self.df.shape[0]} samples in the set and {self.df.shape[1]} columns with names:\n\n{self.df.columns}."
            + tar
            + f"\n\nThe data has the following characteristics:\n{self.df.describe()}"
        )
        return ret

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f"{cls}(df={self.df!r}, target_labels={self.target_labels!r})"

    def describe(self) -> None:
        """Calls __str__ method for analyzer class to print basic information about the dataset."""
        print(self)

    def set_seed(self, seed: int | float) -> None:
        """Allows manual setting of random seed to be used by methods in the Analyzer class.

        Args:
            seed (int | float): Desired seed.
        """
        type_assertion(seed, int | float)
        self.seed: int | float = seed

    def set_target_labels(self, target_labels: str | List[str]) -> None:
        """Sets which column(s) of the self.df are the target.

        Args:
            target_labels (str | List[str]): String or list of strings corresponding to the columns which are to be set as the target.

        Raises:
            IndexError: If a target label does not match an column label from self.df.
        """
        type_assertion(target_labels, str | list)
        if isinstance(target_labels, str):
            try:
                assert target_labels in self.df.columns
            except:
                raise IndexError(
                    "Column name must be found in column names of self.df."
                )
            self.target_labels = [target_labels]
        else:
            type_assertion(target_labels, str, iterable=True)
            try:
                for name in target_labels:
                    assert name in self.df.columns
            except:
                IndexError("Column names must all be found in column names of self.df.")
            self.target_labels = list(target_labels)
        self.n_targets = len(self.target_labels)

    # Dataframe Manipulation Methods
    def get_frame(self) -> pd.DataFrame:
        """Returns self.df

        Returns:
            pd.DataFrame: Current copy of self.df.
        """
        return self.df

    def get_sample_frames(self, return_keys: bool = False) -> None | List[str]:
        # maybe rename to something like describe_sample_frames() and have second function for accessing?
        """Prints the list of keys for and shape of each DataFrame in self.sample_dfs.

        Args:
            return_keys (bool, optional): If set to true, the function will return the set of keys for self.sample_dfs.. Defaults to False.

        Returns:
            List[str]: If return_keys is set to True, a list of the keys will be returned.
        """
        if len(self.sample_dfs) == 0:
            print("There are no frames sampled at this time.")
        else:
            type_assertion(return_keys, bool)
            for i, k in enumerate(self.sample_dfs):
                print(f"{i+1}. [{k}] Shape:{self.sample_dfs[k].shape}")
            if return_keys is True:
                return list(self.sample_dfs.keys())

    def save_frame(
        self,
        save_as: str = "pkl",
        path: str = None,
        sample_df: str = None,
        cols_to_save: List[str] | pd.Index | pd.Series | np.ndarray = None,
    ) -> None:
        """Saves the current frame or one saved to the sample dictionary to a pickle or .csv file.

        Args:
            save_as (str, optional): Type of file the frame is to be saves as from ["pkl", "csv"]. Defaults to "pkl".
            path (str, optional): Directory path at which the file is to be saved. If None is pased, self.dir_path will be used. Defaults to None.
            sample_df (str, optional): If passed, specifies a sampled frame to be used instead of self.df. Defaults to None.
            cols_to_save (List[str] | pd.Index | pd.Series | np.ndarray, optional): Optionable way to save only sepcific columns by passing their labels in an iterable. Defaults to None.

        Raises:
            KeyError: If the save_as option is not one of the acceptable types, or if a non-existant sample_df is called.
        """
        save_options = {"plk": 0, "csv": 1}
        type_assertion(save_as, str), type_assertion(path, str)
        try:
            assert save_as in save_options.keys
        except:
            raise KeyError(f"The str save_as must be one of {save_options.keys}.")
        if path is None:
            path = self.dir_path
        if sample_df is not None:
            type_assertion(sample_df, str)
            try:
                assert sample_df in self.sample_dfs.keys
            except:
                raise KeyError(
                    f"The str for sample_df must correspond to one of the keys in the sample_dfs dictionary: \n{self.get_sample_frames()}"
                )
        else:
            input_df = self.df
        if save_options[save_as] == 0:
            input_df.to_pickle(path)
        elif save_options[save_as] == 1:
            input_df.to_csv(path)
        pass

    def shuffle(self, seed: int | float = None) -> None:
        """Randomly shuffles rows in self.df, with optional seed specifiaction.

        Args:
            seed (int | float, optional): Seed to be used for the shuffling randomization. If not passed, self.seed will be used. Defaults to None.
        """
        if seed is not None:
            type_assertion(seed, int | float)
        else:
            seed = self.seed
        input_df = self.df
        output_df = input_df.sample(frac=1, axis=0, random_state=seed)
        self.df = output_df

        """
        Samples portion of dataset using the pandas df.sample method. Saves to self.sample_dfs dictionary under key=s_name and returns the sampled DataFrame.
        Parameters:
            n (int):
            frac (float):
            seed (int|float,default=self.seed): Optional override of class based seed.
            s_name (str): Optional str to be used as key in sample dictionary
        """

    def sample(
        self,
        n: int = 1,
        frac: float = None,
        seed: int | float = None,
        s_name: str = None,
    ) -> pd.DataFrame:
        """Samples portion of dataset using the pandas df.sample method. Saves to self.sample_dfs dictionary under key=s_name and returns the sampled DataFrame.

        Args:
            n (int, optional): Number of samples to be taken, Cannot be used with frac. Defaults to 1.
            frac (float, optional): Fraction of samples from full set to be taken. Cannot be used with frac. Defaults to None.
            seed (int | float, optional): Optional override of class based seed.. Defaults to None.
            s_name (str, optional): Optional str to be used as key in sample dictionary. If not passed, the key will be automaticaly named with the following format "sampleFrame_{n_sampleFrames in self.samples}." Defaults to None.

        Raises:
            ValueError: The value for frac must be in the range (0,1].

        Returns:
            pd.DataFrame: Sampled dataframe.
        """
        if seed is not None:
            type_assertion(seed, int | float)
        else:
            seed = self.seed
        if s_name is None:
            s_name = f"sampleFrame_{len(self.sample_dfs.keys())+1}"
        else:
            type_assertion(s_name, str)
        input_df = self.df
        if frac is not None:
            type_assertion(frac, float)
            try:
                assert frac > 0 and frac <= 1
            except:
                raise ValueError("Value for frac must be in range 0<x<=1.")
            output_df: pd.DataFrame = input_df.sample(
                frac=frac, random_state=seed, axis=0
            )
        else:
            type_assertion(n, int)
            try:
                assert n > 0 and n <= self.n_samples
            except:
                raise ValueError(
                    f"Value for n must be positive and not greater than the total number of samples ({self.n_samples})"
                )
            output_df: pd.DataFrame = input_df.sample(n=n, random_state=seed, axis=0)
        self.sample_dfs[s_name] = output_df
        return output_df

    def drop_columns(self, cols_to_drop: List) -> None:
        """
        Takes in list of column labels to be dropped and drops from them from the self.df.
        Parameters:
            cols_to_drop (List): List of column labels to be dropped from the input_df frame.
        """
        type_assertion(cols_to_drop, list)
        input_df = self.df
        try:
            assert len(cols_to_drop) > 0
            assert set([x in list(input_df.columns) for x in cols_to_drop]) == {True}
        except:
            raise IndexError(
                "List must contain only existing column names in the input_df and must not be empty."
            )
        output_df = input_df.drop(cols_to_drop, axis=1)
        self.df = output_df
        self.n_features = len(self.df.columns) - self.n_targets

    # Data Cleaning Methods (str cleaning?)
    def drop_missing_data(
        self, cols_to_check: List[str] = None, percentage_threshold: float = 0.1
    ) -> None:
        """
        Calculates the percentage of missing values of self.df and removes all missing values if below the pecrentage_threshold.
        Parameters:
            cols_to_check (List[str], default=None): Optional list of column keys to specify which columns to check for missing values. If None, all columns will be checked.
            percentage_threshold (float (0<=x<1), default=0.1): Threshold of missing values which will raise an error due to dataloss.
        """
        input_df = self.df
        type_assertion(input_df, pd.DataFrame), type_assertion(
            percentage_threshold, float
        )
        try:
            assert percentage_threshold >= 0 and percentage_threshold < 1
        except:
            raise ValueError("The percentage_threshold must be in range 0<=x<1.")
        if cols_to_check is None:
            cols_to_check = list(input_df.columns)
        n_samples: int = input_df.shape[0]
        missing_count_by_row: pd.Series = (
            input_df[cols_to_check].isna().max(axis=1).sum()
        )
        missing_perc: pd.Series = missing_count_by_row / n_samples
        if missing_perc.max() > percentage_threshold:
            raise ValueError(
                f"The number of missing values in the dataset exceeds the threshold of {percentage_threshold}."
            )
        output: pd.DataFrame = (
            input_df[cols_to_check]
            .dropna()
            .merge(input_df[~cols_to_check], how="left", on=input_df.index)
        )
        self.df: pd.DataFrame = output
        self.n_samples: int = self.df.shape[0]

    def fill_missing_data(
        self,
        fill: str | int | float | dict | pd.Series | pd.DataFrame | None = ...,
        cols_to_fill: List[str] = None,
    ) -> None:
        """
        Fills and overwrittes missing values of self.df.
        Parameters:
            fill (object): Str or numeric value to fill missing values with. If fill is a str in ['backfill', 'bfill', 'ffill'], the corresponsing fill method from pandas fillna() will be used.
            cols_to_fill (List(str), default=None): Optional list of column labels which will be filled. If not passed, all collumns will be filled.
        """
        # better error protection for fill?
        if cols_to_fill is None:
            cols_to_fill = self.df.columns
        input_df: pd.DataFrame = self.df[cols_to_fill]
        if fill in ["backfill", "bfill", "ffill"]:
            output_df = input_df.fillna(method=fill)
        else:
            output_df = input_df.fillna(value=fill)
        self.df = pd.concat(output_df, input_df.drop(cols_to_fill, axis=1))

    def set_col_dtype(self, col: str, dtype: str) -> None:
        """
        Sets the dtype of specified column.
        Parameters:
            col (str):
            dtype (type):
        """
        type_assertion(col, str)
        try:
            assert col in self.df.columns
        except:
            raise ValueError(
                "The variable col must be a str corresponding to a column in self.df."
            )
        self.df[col] = self.df[col].astype(dtype=dtype)

    # Data Preprocessing Methods (create_feature method: adds new column based on functions such as two columns added or multiplied?)
    def encode_features(
        self, encoder: str, cols_to_encode: List[str] = None, **encoderkwargs
    ) -> None:
        """
        Takes in unencoded data set and encodes categorical features with either OrdinalEncoder or OneHotEncoder from sklearn as specified.
        Parameters:
            encoder (str): Sklearn encoder to be used for encoding. {'ohe': OneHotEncoder(), 'ord': OrdinalEncoder}.
            cols_to_encode (List[str], default=None): Optional list of column names to be encoded. If not passed, all applicable columns will be encoded.
        """
        type_assertion(encoder, str), type_assertion(cols_to_encode, str, iterable=True)
        encoders: dict = {"ohe": OneHotEncoder, "ord": OrdinalEncoder}
        try:
            assert encoder in encoders.keys()
        except:
            raise ValueError("Value for encoder must be either 'ohe' or 'ord'.")
        enc = encoders[encoder](**encoderkwargs)
        if cols_to_encode is None:
            cols_to_encode = self.df.columns
        if self.target_labels is not None and self.target_labels in cols_to_encode:
            cols_to_encode = cols_to_encode.drop(self.target_labels, axis=1)
        for col in cols_to_encode:
            self.df.loc[:, col] = enc.fit_transform(self.df.loc[:, [col]])

    def encode_target(
        self, encoder: str, set_target_labels: str = None, **encoderkwargs
    ) -> None:
        """
        Takes in unencoded target set and encodes categorical features with either OrdinalEncoder or OneHotEncoder from sklearn as specified.
        Parameters:
            encoder (str): String representing the desired sklearn encoder to use. {'lab': LabelEncoder, 'ohe': OneHotEncoder, 'ord': OrdinalEncoder}
            set_target_labels (str, default=None): Optional target designation, must be string of column name.
        """
        type_assertion(encoder, str)
        encoders: dict = {
            "lab": LabelEncoder,
            "ohe": OneHotEncoder,
            "ord": OrdinalEncoder,
        }
        try:
            assert encoder in encoders.keys()
        except:
            raise ValueError("Encoder must be str value of 'lab' or 'ohe'.")
        if set_target_labels is not None:
            type_assertion(set_target_labels, str)
            self.set_target_labels(set_target_labels)
        elif self.n_targets == 0:
            raise ValueError("No target set, please set target and try again.")
        enc = encoders[encoder](**encoderkwargs).set_output(transform="pandas")
        target = self.df[self.target_labels]
        # Need to test if the above works or if the n_targets==1 case needs special treatment
        # if self.n_targets == 1:
        #     target = self.df[self.target_labels]
        # else:
        #     target = self.df[self.target_labels]
        target = enc.fit_transform(target)
        new_target_labs = target.columns
        self.df = pd.concat([self.df.drop(self.target_labels, axis=1), target], axis=1)
        self.target_labels = list(new_target_labs)

    def scale(
        self,
        scaler: str = "sta",
        scale_target: bool = True,
        cols_to_scale: List[str] = None,
        **scalerkwargs,
    ) -> None:
        """
        Scales columns in self.df with chosen scaler
        Parameters:
            scaler (str, default='sta'): Choice of sklearn scaler to use: {'sta': StandardScaler, 'minmax': MinMaxScaler, 'robust': RobustScaler}.
            scale_target (bool, default=True): Allows exclusion of target column from scaling.
            cols_to_scale: Optional list of columns to be scaled If not passed all applicable columns will be scaled.
            minmax_range (tuple, default=(0,1)): Range to use if MinMaxScaler is used.
        """
        type_assertion(scaler, str), type_assertion(scale_target, bool)
        if cols_to_scale is not None:
            type_assertion(cols_to_scale, str, iterable=True)
        scalers = {
            "sta": StandardScaler,
            "minmax": MinMaxScaler,
            "robust": RobustScaler,
        }
        try:
            assert scaler in scalers.keys()
        except:
            raise ValueError("Value for scaler must be 'sta', 'minmax' or 'robust'.")
        sca = scalers[scaler](**scalerkwargs).set_output(transform="pandas")
        if cols_to_scale is None:
            cols_to_scale = self.df.columns
        if scale_target != True:
            cols_to_scale = cols_to_scale.drop(self.target_labels)
        scaled_df = self.df.loc[:, cols_to_scale]
        scaled_df = sca.fit_transform(scaled_df)
        output_df = pd.concat((self.df.drop(cols_to_scale, axis=1), scaled_df), axis=1)
        self.df = output_df

    # Exploritory and Feature Selection Methods
    def calc_corr_to_target(self, return_targets: bool = False) -> pd.DataFrame:
        """
        Calculates and returns array of feature by target correlation.
        Parameters:
            return_targets (bool, default=False): Allows the returning of the target columns in the correlation 0 axis.
        Returns:
            pd.DataFrame: Feature by target correlation matrix

        """
        type_assertion(return_targets, bool)
        if return_targets is False:
            target_corr = self.df.corr()[self.target_labels].drop(
                self.target_labels, axis=0
            )
            return target_corr
        else:
            target_corr = self.df.corr()[self.target_labels]
            return target_corr

    def calc_corr_matrix(self, include_target: bool = False) -> pd.DataFrame:
        """
        Calculates and returns array of feature by feature correlation.
        Parameters:
            inlcude (bool, default=False): Allows the inclusion of the target columns in the matrix.
        Returns:
            pd.DataFrame: Feature by feature correlation matrix
        """
        type_assertion(include_target, bool)
        input_df = self.df
        if include_target is False:
            corr = input_df.drop(labels=self.target_labels, axis=1).corr()
            return corr
        else:
            corr = input_df.corr()
            return corr

    def cut_features_by_corr(
        self,
        t_corr_threshold: float = 0.0,
        f_corr_threshold: float = None,
        comb_threshold: float = None,
        num_to_cut: int = None,
    ) -> None:
        """
        Drops feature columns based on correlation. This can be set by threshold amount or by manually setting the number of columns to cut.
        Parameters:
            t_corr_threshold (float, default=0.): Value to be used as threshold for cutting features based on their correlation to the target. The features less than this threshold will be cut. The t_corr can range from -1 to 1. Threshold will be overwritten if num_to_cut!=None or if it is set to None.
            f_corr_threshold (float, default=None): Value to be used as threshold for cutting features based on their mean correlation to other features. The features greater than this threshold will be cut. The f_corr can range from -1 to 1.
            comb_threshold (float, default=None): Value to be used as threshold for cutting features based on their combines t_corr and f_corr. The formula for this metric is: (t_corr + 1) * (1 - f_corr) / 4. This value can range from 0 to 1 with higher value representing a more meaningful feature in the set.
            num_to_cut (int, default=None): Value to be used if a specific number of columns is desired to be dropped. The values with the lowest comb_corr score will be dropped in ascending order until value passed to num_to_cut is reached.
        """
        # consider adding choice over aggregation method for f_corr
        # also consider iterative cutting of f_corr since the aggregate correlation will change with each dropped column as colinearity decreases
        type_assertion(t_corr_threshold, float)
        t_corr: pd.DataFrame = self.calc_corr_to_target()
        if t_corr.shape[1] > 1:
            t_corr = t_corr.max(axis=1)
        f_corr: pd.DataFrame = self.calc_corr_matrix()
        corr_df = pd.DataFrame()
        corr_df["t_corr"] = t_corr  # aim to maximize
        corr_df["f_corr"] = f_corr.mean()  # aim to minimize
        corr_df["comb_corr"] = (
            (corr_df["t_corr"].abs() + 1) * (1 - corr_df["f_corr"].abs()) / 4
        )  # combined metric, should range 0, 1. aim to maximize
        corr_df.sort_values(by="comb_corr", axis=0, ascending=False, inplace=True)
        if num_to_cut is not None:
            type_assertion(num_to_cut, int)
            try:
                assert num_to_cut > 0 and num_to_cut < self.n_features
            except:
                raise ValueError(
                    f"Value of num_to_cut must be greater than 0 and less than the number of features ({self.n_features})."
                )
            cols_to_cut = corr_df.index[-num_to_cut:]
            self.drop_columns(cols_to_cut)
        else:
            if t_corr_threshold is not None and corr_df.shape[0] > 0:
                type_assertion(t_corr_threshold, float)
                cut_df = corr_df.loc[
                    corr_df["t_corr"].abs() < t_corr_threshold
                ]  # leaves in columns failing threshold values
            if f_corr_threshold is not None and cut_df.shape[0] > 0:
                type_assertion(f_corr_threshold, float)
                cut_df = corr_df.loc[corr_df["f_corr"].abs() > f_corr_threshold]
            if comb_threshold is not None and cut_df.shape > 0:
                type_assertion(comb_threshold, float)
                cut.df = corr_df.loc[corr_df["comb_corr"] < comb_threshold]
            cols_to_cut = list(cut_df.index)
            self.drop_columns(cols_to_cut)

    # Plotting Methods
    def plot_correlationMatrix(
        self,
        cols_to_plot: List[str] = None,
        display_plot: bool = True,
        save_plot: bool = False,
        save_path: str = None,
        include_target: bool = False,
    ) -> None:
        """
        Calculates then optionally displays and saves annotated correlation matrix plo of self.df.
        Parameters:
            cols_to_plot (List[str], default=None): Optional parameter to specify which columns to be included. If left as None, all columns will be plotted.
            display_plot (bool, default=True): Boolean specifying if image should be displayed.
            save_plot (bool, default=True): Boolean specifying if a image of the plot will be saved to disk.
            save_path (str, default=''): path of directory where the plot will be saves.
            include_target (bool, default=False): Optional boolean to include target in the plot.
        """
        input_df = self.df
        if save_path == None:
            save_path = self.dir_path
        if cols_to_plot is None:
            try:
                for name in cols_to_plot:
                    assert name in input_df
            except:
                IndexError("Column names must all be found in column names of self.df.")
            cols_to_plot = self.df.columns
        if include_target is False:
            input_df = input_df.drop(self.target_labels, axis=1)

        corr = input_df[cols_to_plot].corr()
        fig = sns.heatmap(corr, annot=True, center=0).figure
        if save_plot is True:
            fig.savefig(save_path + "corr_matrix.png", bbox_inches="tight", format="png")
        if display_plot is True:
            fig.show()
        plt.close(fig)

    def plot_histograms_numeric(
        self,
        cols_to_plot: List[str] = None,
        display_plot: bool = True,
        save_plot: bool = False,
        save_path: str = None,
        include_target: bool = False,
        plot_grid_width: int = 3,
    ) -> None:
        """
        Plots and optionally displays and saves histogram plots of each numeric feature.
        Parameters:
            cols_to_plot (List[str], default=None): Optional parameter to specify which columns to be included. If left as None, all columns will be plotted.
            display_plot (bool, default=True): Boolean specifying if image should be displayed.
            save_plot (bool, default=True): Boolean specifying if a image of the plot will be saved to disk.
            save_path (str, default=''): path of directory where the plot will be saves.
            include_target (bool, default=False): Optional boolean to include target (if numeric) in the plot.
            plot_grid_width (int, deafult=3):
        """
        if save_path == None:
            save_path = self.dir_path
        type_assertion(display_plot, bool), type_assertion(
            save_plot, bool
        ), type_assertion(save_path, str), type_assertion(
            include_target, bool
        ), type_assertion(
            plot_grid_width, int
        )
        input_df: pd.DataFrame = self.df.select_dtypes(include="number")
        if cols_to_plot is None:
            cols_to_plot = input_df.columns
        else:
            try:
                for name in cols_to_plot:
                    assert name in input_df.columns
            except:
                IndexError(
                    "Column names must all be found in column names of self.df and must be only for numeric columns."
                )
        if include_target is False and len(self.target_labels) > 0:
            input_df = input_df.drop(self.target_labels, axis=1)
            for lab in self.target_labels:
                cols_to_plot = list(cols_to_plot)
                if lab in cols_to_plot:
                    cols_to_plot.remove(lab)
        try:
            assert len(cols_to_plot) > 0
        except:
            raise IndexError("No features to plot with these parameters.")
        plot_grid_width = min(len(cols_to_plot), plot_grid_width)
        feat_list = list(cols_to_plot)
        n_features_to_plot: int = len(feat_list)
        n_cols: int = plot_grid_width
        n_rows: int = math.ceil(n_features_to_plot / plot_grid_width)
        plt.figure(figsize=(n_cols * 5, n_rows * 5))
        if include_target is True:
            targets = input_df[self.target_labels[0]].unique()
            for i, feat_label in enumerate(feat_list):
                plt.subplot(n_rows, n_cols, i + 1)
                # feat = plot_df[feat_label]
                # Plot histograms for each target class
                for x in targets:
                    plt.hist(
                        input_df.loc[input_df[self.target_labels[0]] == x, feat_label],
                        alpha=0.5,
                        label=str(x),
                    )
                plt.title(f"{feat_label}")
                plt.legend()
        else:
            for i, feat_label in enumerate(feat_list):
                if i + 0 > n_features_to_plot:
                    pass
                else:
                    plt.subplot(n_rows, n_cols, i + 1)
                    feat = input_df[feat_label]
                    plt.hist(feat)
                    plt.title(f"{feat_label}")
        plt.suptitle(f"Numeric Feature Histograms (n={self.n_samples})")
        plt.tight_layout()
        if save_plot is True:
            plt.savefig(save_path + "numeric_histograms.png", format="png")
        if display_plot is True:
            plt.show()
        plt.close()


    def plot_histograms_categorical(
        self,
        cols_to_plot: List[str] = None,
        display_plot: bool = True,
        save_plot: bool = False,
        save_path: str = None,
        include_target: bool = False,
        plot_grid_width: int = 3,
    ) -> None:
        """
        Plots and optionally displays and saves histogram plots of each categorical feature.
        Parameters:
            cols_to_plot (List[str], default=None): Optional parameter to specify which columns to be included. If left as None, all columns will be plotted.
            display_plot (bool, default=True): Boolean specifying if image should be displayed.
            save_plot (bool, default=True): Boolean specifying if a image of the plot will be saved to disk.
            save_path (str, default=''): path of directory where the plot will be saves.
            include_target (bool, default=False): Optional boolean to include target (if numeric) in the plot.
            plot_grid_width (int, deafult=3):
        """
        if save_path == None:
            save_path = self.dir_path
        type_assertion(display_plot, bool), type_assertion(
            save_plot, bool
        ), type_assertion(save_path, str), type_assertion(
            include_target, bool
        ), type_assertion(
            plot_grid_width, int
        )
        input_df: pd.DataFrame = self.df.select_dtypes(
            include=["category", "string", "bool"]
        )
        if cols_to_plot is None:
            cols_to_plot = input_df.columns
        else:
            try:
                for name in cols_to_plot:
                    assert name in input_df.columns
            except:
                IndexError(
                    "Column names must all be found in column names of self.df and must be only for categorical columns."
                )
        if include_target is False and len(self.target_labels) > 0:
            drop = False
            for target in self.target_labels:
                if target in list(input_df.columns):
                    drop = True
            if drop is True:
                input_df = input_df.drop(self.target_labels, axis=1)
            for lab in self.target_labels:
                cols_to_plot = list(cols_to_plot)
                if lab in cols_to_plot:
                    cols_to_plot.remove(lab)
        try:
            assert len(cols_to_plot) > 0
        except:
            raise IndexError("No features to plot with these parameters.")
        plot_grid_width = min(len(cols_to_plot), plot_grid_width)
        feat_list = list(cols_to_plot)
        n_features_to_plot: int = len(feat_list)
        n_cols: int = plot_grid_width
        n_rows: int = math.ceil(n_features_to_plot / plot_grid_width)
        plt.figure(figsize=(n_cols * 5, n_rows * 5))

        if include_target is True:
            targets = input_df[self.target_labels[0]].unique()
            for i, feat_label in enumerate(feat_list):
                plt.subplot(n_rows, n_cols, i + 1)
                # feat = plot_df[feat_label]
                # Plot histograms for each target class
                for x in targets:
                    plt.hist(
                        input_df.loc[input_df[self.target_labels[0]] == x, feat_label],
                        alpha=0.5,
                        label=str(x),
                    )
                plt.title(f"{feat_label}")
                plt.legend()
        else:
            for i, feat_label in enumerate(feat_list):
                if i + 0 > n_features_to_plot:
                    pass
                else:
                    plt.subplot(n_rows, n_cols, i + 1)
                    feat = input_df[feat_label]
                    plt.hist(feat)
                    plt.title(f"{feat_label}")
        plt.suptitle(f"Categorical Feature Histograms (n={self.n_samples})")
        plt.tight_layout()
        if save_plot is True:
            plt.savefig(save_path + "categorical_histograms.png", format="png")
        if display_plot is True:
            plt.show()
        plt.close()

    def plot_boxPlot(
        self,
        cols_to_plot: List[str] = None,
        display_plot: bool = True,
        save_plot: bool = False,
        save_path: str = None,
    ) -> None:
        """
        Plots and optionally displays and saves box plots of features.
        Parameters:
            cols_to_plot (List[str], default=None): Optional parameter to specify which columns to be included. If left as None, all columns will be plotted.
            display_plot (bool, default=True): Boolean specifying if image should be displayed.
            save_plot (bool, default=False): Boolean specifying if a image of the plot will be saved to disk.
            save_path (str, default=''): path of directory where the plot will be saves.
        """
        input_df: pd.DataFrame = self.df
        if save_path == None:
            save_path = self.dir_path
        if cols_to_plot is None:
            cols_to_plot = input_df.columns
        else:
            try:
                for name in cols_to_plot:
                    assert name in input_df.columns
            except:
                IndexError(
                    "Column names must all be found in column names of self.df and must be only for numeric columns."
                )
        input_df = input_df[cols_to_plot]
        sns_plt = sns.boxplot(input_df, palette="muted")
        sns_plt.tick_params(axis="x", rotation=20)
        fig = sns_plt.get_figure()
        fig.suptitle(f"Boxplots of feature distributions (n = {self.n_samples})")
        if save_plot is True:
            fig.savefig(save_path + "boxPlot.png", bbox_inches="tight", format="png")
        if display_plot is True:
            fig.show()
        plt.close(fig)

    def plot_pairPlot(
        self,
        cols_to_plot: List[str] = None,
        display_plot: bool = True,
        save_plot: bool = False,
        save_path: str = None,
        include_target: bool = False,
    ) -> None:
        """
        Plots and optionally displays and saves pairplot of features.
        Parameters:
            cols_to_plot (List[str], default=None): Optional parameter to specify which columns to be included. If left as None, all columns will be plotted.
            save_plot (bool, default=False): Boolean specifying if a image of the plot will be saved to disk.
            save_path (str, default=''): path of directory where the plot will be saves.
            include_target (bool, default=False): Boolean specifying if the target (only first is used if multiple targets exist) is included in the plot.
        """
        input_df: pd.DataFrame = self.df
        if save_path == None:
            save_path = self.dir_path
        if cols_to_plot is None:
            cols_to_plot = input_df.columns
        else:
            try:
                for name in cols_to_plot:
                    assert name in input_df.columns
            except:
                IndexError(
                    "Column names must all be found in column names of self.df and must be only for numeric columns."
                )
        input_df = input_df[cols_to_plot]
        if include_target is True:
            sns_plt = sns.pairplot(input_df, hue=self.target_labels[0], palette="muted")
        else:
            sns_plt = sns.pairplot(input_df)
        fig = sns_plt.figure
        if save_plot is True:
            fig.savefig(save_path + "pairPlot.png", bbox_inches="tight", format="png")
        if display_plot is True:
            fig.show()
        plt.close(fig)





# def feature_select(input_X: pd.DataFrame, input_y: pd.Series, method: str='auto', feature_list: List[str]=None):
#     """
#     Determines and selects the features to used in the final model based on correlation threshold to target and other features.
#     Parameters:
#         input_X (pd.DataFrame): DataFrame of features
#         input_y (pd.Series): Series of target variable
#         method (str): Method for determining features to use from options ['auto', 'manual']. If 'manual' is passed, a list of features to include must be passed as 'feature_list'
#         feature_list (List[str]): List of features to include if 'manual' method is used. Items in list must be str that correspond to column names.
#     """
#     input_X['target'] = input_y
#     corr_matrix = input_X.corr()
#     target_corr = corr_matrix['target']
# def split(*input: pd.DataFrame|pd.Series|np.ndarray, split_amts: tuple) -> pd.DataFrame|pd.Series|np.ndarray:
#     """
#     Splits the dataset into Train, Test, and Validation sets with specified proportions.
#     Parameters:
#         input (pd.DataFrame | pd.Series | np.ndarray | List): Encoded and Scaled Dataframe(s), Arrays and/or Series to be split.
#         split_amts (tuple): Tuple describing the percentages of dataset to allocate to test and validation splits. The format is a tuple of length 2: (portion to allocate to test set, portion to allocate to validation set). Ex: split(input, split_amts = (0.2, 0.1)) would result in splitting the set into train = 70%, test = 20% and validation = 10%.
#     Returns:
#         pd.DataFrame: Several dataframes split to specification with train_test_split from sklearn.
#     """
#     try:
#         assert type(input) == pd.DataFrame or type(input) == pd.Series or type(input) == np.ndarray or type(input) == List
#     except:
#         raise TypeError('The input must be a Dataframe, Series, numpy array, or list of such.')
#     try:
#         assert type(split_amts) == tuple and len(split_amts) == 2 and sum(split_amts) <= 1 and split_amts[0] >= 0 and split_amts[1] >= 0
#     except:
#         raise TypeError('The input must be a tuple of length 2 containing two valid, positive float values representing the desired portions to split into test and validation sets.')
#     else:
#         if type(input) == List or type(input) == tuple:
#             for array in input:
#                 try:
#                     assert type(array) == pd.DataFrame or type(array) == pd.Series or type(array) == np.ndarray
#                 except:
#                     raise TypeError('The items in the input list must be of type Dataframe, Series, or numpy array.')

#     test_p: float = split_amts[0]
#     val_p: float = split_amts[1]

#     split_size1 = test_p + val_p
#     split_size2 = val_p / (val_p + test_p)

#     *splits, = train_test_split(input,test_size=split_size1)
#     *splits, = train_test_split(input,test_size=split_size2)
