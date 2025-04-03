import os
import numpy as np
import pandas as pd
from typing import List, Dict
import optuna
import pickle as pkl
import random
from abc import ABC, abstractmethod

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from tensorflow.random import set_seed
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session
import tensorflow.keras.saving as ks
import keras_tuner as kt

import matplotlib.pyplot as plt

from utils.utils import (
    type_assertion,
    save_to_JSON,
    load_JSON,
    extension_extract,
    create_optuna_suggestion,
    create_kt_suggestion,
    create_param_dict,
)


available_models = {
    "lin_reg": Ridge,
    "KNN": KNeighborsRegressor,
    "decision_tree": DecisionTreeRegressor,
    "random_forest": RandomForestRegressor,
    "SVR": SVR,
    "XG_boost": XGBRegressor,
    "ANN": "ANN",
}

metric_fns = {
    "mean_squared_error": mean_squared_error,
    "mean_absolute_error": mean_absolute_error,
    "r2_score": r2_score,
    "root_mean_squared_error": root_mean_squared_error,
}


def compile_keras_regression_model(
    hp,
    input_shape: np.ndarray,
    hyperparam_grid: dict,
):
    """Compiles a keras model to be pased to a keras_tuner tuner for hyperparameter tuning. This function is only passed by a tuner object.

    Args:
        hp: Hyperparameter object for keras-tuner.
        input_shape (np.ndarray): Input shape of a single observation.
        hyperparam_grid (dict): Dictionary of hyperparameters to search across
    """

    model = Sequential()

    n_dense_layers = create_kt_suggestion(
        hp, model_type="ANN", parameter="n_dense_layers", param_dict=hyperparam_grid
    )
    dense_units = create_kt_suggestion(
        hp, model_type="ANN", parameter="dense_units", param_dict=hyperparam_grid
    )
    dropout_rate = create_kt_suggestion(
        hp, model_type="ANN", parameter="dropout_rate", param_dict=hyperparam_grid
    )
    learning_rate = create_kt_suggestion(
        hp, model_type="ANN", parameter="learning_rate", param_dict=hyperparam_grid
    )
    # ^ is there a more generalizable way to do this?

    optimizer = Adam(learning_rate=learning_rate)

    for i in range(0, n_dense_layers):
        if i == 0:
            model.add(Dense(units=dense_units, activation="relu", input_shape=input_shape))
            model.add(Dropout(rate=dropout_rate))
        else:
            model.add(Dense(units=dense_units, activation="relu"))
            model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=1))

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[
            MeanAbsoluteError(name="mae"), 
            MeanSquaredError(name="mse"),
            ],
    )
    return model


def compile_keras_regression_model_manual(
    n_layers: int,
    dense_units: int,
    dropout_rate: float,
    learning_rate: float,
    activation: str = "relu",
) -> Sequential:
    """Function allowing a manual creation of a keras ANN model with specified hyperparameters.

    Args:
        n_layers (int): Desired number of fully connected layers for the model.
        dense_units (int): Number of units per dense layer.
        dropout_rate (float): Desired dropout rate. Must be in range [0,1).
        learning_rate (float): Desired learning rate to be used by the optimizer.
        activation (str, optional): Activation function to use. Limited to: ["relu", "leaky_relu", "tanh", "sigmoid", "softmax"]. Defaults to "relu".

    Raises:
        ValueError: If any of the arguements are invalid values.

    Returns:
        Sequential: Keras Sequential() model with specified hyperparameters.
    """
    type_assertion(n_layers, int), type_assertion(dense_units, int), type_assertion(
        dropout_rate, float | int
    ), type_assertion(learning_rate, float | int)
    try:
        assert (
            n_layers > 0 and dense_units > 0 and dropout_rate > 0 and learning_rate > 0
        )
    except:
        raise ValueError(
            "The values for n_layers, dense_units, dropout_rate, and learning_rate must be strictly positive."
        )
    try:
        assert activation in ["relu", "leaky_relu", "tanh", "sigmoid", "softmax"]
    except:
        raise ValueError(
            "The value for activation must be one of: ['relu', 'leaky_relu', 'tanh', 'sigmoid', 'softmax']"
        )

    model = Sequential
    optimizer = Adam(learning_rate=learning_rate)

    for layer in n_layers:
        model.add(Dense(units=dense_units, activation=activation))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=1))
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[
            MeanAbsoluteError(name="mae"), 
            MeanSquaredError(name="mse"),
            ],
    )
    return model


class RegressionModel(ABC):
    """
    Abstract Class for generic regression model.
    """

    def __init__(self, seed: int | float = 0, model: object = None, **params):
        """Contructor basis for all Regression models.

        Args:
            seed (int | float, optional): Seed to set for all randomized actions taken by the class. Defaults to 0.
            model (object, optional): Model Object. Defaults to None.
        """
        self.model = model
        self.model_type: str = None
        self.seed: int | float = seed
        self.additional_model_params: dict = params
        self.fitted_model = None
        self.metrics = {}
        self.hyperparams = None
        self.n_features = None

    def __str__(self) -> str:
        """Returns str containing basic information about the model.

        Returns:
            str: Basic information about the model.
        """
        if self.fitted_model is not None:
            if len(self.metrics.keys()) > 0:
                return (
                    f"This is a model of type: {self.model_type}."
                    + f"It has been fit with the following hyperparameters: {self.hyperparams}."
                    + f"It has the following performance metrics: {self.metrics}."
                )
            else:
                return (
                    f"This is a model of type: {self.model_type}."
                    + f"It has been fit with the following hyperparameters: {self.hyperparams}."
                )
        else:
            return f"This is a model of type: {self.model_type}."

    @abstractmethod
    def describe(self) -> None:
        """Prints basic information about the model."""

    def __call__(
        self,
        trial,
        x: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        hyperparam_grid: dict,
        metric_choice: str = "mean_squared_error",
        **params,
    ) -> float:
        """Method for use only by optuna trials to choose hyperparams from grid, fit model, and return metric.

        Args:
            trial: Optuna trial
            x (pd.DataFrame): Feature dataset.
            y (pd.Series | pd.DataFrame): Target dataset.
            hyperparam_grid (dict): Hyperparameter dictionary to specify gird search space.
            metric_choice (str, optional): Metric to optimize hyperparameters for from: ["mean_squared_error", "mean_absolute_error", "r2_score", "root_mean_squared_error"]. Defaults to "f1".

        Raises:
            ValueError: If a non-valid str is passed for metric_choice.

        Returns:
            float: Specified metric which was achieved by the model in this trial.
        """

        if self.model_type != "ANN":
            try:
                assert metric_choice in ["mean_squared_error", "mean_absolute_error", "r2_score", "root_mean_squared_error"]
            except:
                raise ValueError(
                    "The value for metric_choice must be in ['mean_squared_error', 'mean_absolute_error', 'r2_score', 'root_mean_squared_error']"
                )
            if metric_choice == "r2_score":
                metric_choice = "r2"
            else:
                metric_choice = "neg_" + metric_choice
            params: dict = create_param_dict(
                trial=trial, model_type=self.model_type, hyperparam_grid=hyperparam_grid
            )
            model = self.model(**params)
            model.fit(x, y)
            metric = cross_val_score(model, x, y, scoring=metric_choice, cv=5)
            metric = np.mean(metric)
            return metric

    def save_model(self, path: str, custom_name: str = ""):
        """Saves a model object as a pickle file at the specified path.

        Args:
            path (str): Directory path where the file is to be saved.
            custom_name (str): Name to include in filename. Default = "".

        Raises:
            AttributeError: If the model has not yet been fit, it cannot be saved. This is to prevent saving empty models.
        """
        try:
            assert self.fitted_model is not None
        except:
            raise AttributeError("The model must be fitted before it can be saved.")
        with open(f"{path}{self.model_type}{custom_name}.pkl", "wb") as file:
            pkl.dump(self.fitted_model, file)

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series | pd.DataFrame):
        """Trains model based on input parameters.

        Args:
            x_train (pd.DataFrame): Feature data to be used for training.
            y_train (pd.Series | pd.DataFrame): Target data to be used for training.

        Raises:
            ValueError: If the datasets have a mismatch in number of rows.
        """
        type_assertion(x_train, pd.DataFrame | np.ndarray), type_assertion(
            y_train, pd.Series | pd.DataFrame | np.ndarray
        )
        try:
            assert x_train.shape[0] == y_train.shape[0]
        except:
            raise ValueError("x_train and y_train must have the same number of rows.")
        self.model.fit(x_train, y_train)
        self.hyperparams = self.model.get_params()
        self.fitted_model = self.model
        self.n_features = x_train.shape[1]

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Inputs the dataset x into the trained model and returns the predicted targets.

        Args:
            x (pd.DataFrame): Dataset which the model will predict from. Must have the same feature set as the data which the model was originally trained on.

        Raises:
            AttributeError: If the model has not yet been fit, it cannot be used to predict.
            ValueError: If the number of features in x doesn't match the number of features in the set which was used to train the model.

        Returns:
            np.ndarray: Output of predicted targets.
        """
        try:
            assert self.fitted_model is not None
        except:
            raise AttributeError(
                "The model must be fitted before it can be used to predict."
            )
        type_assertion(x, pd.DataFrame | np.ndarray)
        try:
            assert x.shape[1] == self.n_features
        except:
            raise ValueError(
                f"x must have the same number of feature columns as the data which was used to train it. x has {x.shape[1]} columns where {self.n_features} columns was expected."
            )
        return self.fitted_model.predict(x)

    def score(
        self,
        y_true: pd.Series | pd.DataFrame,
        y_pred: pd.Series | pd.DataFrame = None,
        x: pd.DataFrame = None,
    ) -> Dict[str, float]:
        """Returns performance metrics ("mean_squared_error", "mean_absolute_error", "r2_score", "root_mean_squared_error") for the model.

        Args:
            y_true (pd.Series | pd.DataFrame): True target values to be passed and scored against.
            y_pred (pd.Series | pd.DataFrame, optional): Predicted target values to score. Defaults to None.
            x (pd.DataFrame, optional): DataFrame to be scores upon in comparison to y_true. If passed, value for y_pred will be overwritten and a new fitting will take places using x. Defaults to None.

        Raises:
            ValueError: If neither y_pred or x is passed.

        Returns:
            Dict[str, float]: Dictionary of metric values.
        """
        if y_pred is None:
            try:
                assert x is not None
            except:
                raise ValueError("A value must be passed for either y_pred or x")
        if x is not None:
            y_pred = self.predict(x)
        output = {}
        output["mean_squared_error"] = mean_squared_error(y_true, y_pred)
        output["mean_absolute_error"] = mean_absolute_error(y_true, y_pred)
        output["r2_score"] = r2_score(y_true, y_pred)
        output["root_mean_squared_error"] = root_mean_squared_error(y_true, y_pred)
        return output


class LinearRegression_(RegressionModel):
    """ """

    def __init__(self, seed=0, model=None, **params):
        super().__init__(seed, model, **params)
        if model is None:
            self.model = Ridge
        self.model_type: str = "lin_reg"
        self.seed = seed
        self.fitted_model = None
        self.additional_model_params: dict = params
        self.metrics = {}

    def describe(self) -> None:
        """Prints basic information about the model."""
        print(self)


class KNN_(RegressionModel):
    def __init__(self, seed=0, model=None, **params):
        super().__init__(seed, model, **params)
        if model is None:
            self.model = KNeighborsRegressor
        self.model_type = "KNN"
        self.seed = seed
        self.fitted_model = None
        self.additional_model_params: dict = params
        self.metrics = {}

    def describe(self) -> None:
        """Prints basic information about the model."""
        print(self)


class DecisonTree_(RegressionModel):
    def __init__(self, seed=0, model=None, **params):
        super().__init__(seed, model, **params)
        if model is None:
            self.model = DecisionTreeRegressor
        self.model_type = "decision_tree"
        self.seed = seed
        self.fitted_model = None
        self.additional_model_params: dict = params
        self.metrics = {}

    def describe(self) -> None:
        """Prints basic information about the model."""
        print(self)


class RandomForest_(RegressionModel):
    def __init__(self, seed=0, model=None, **params):
        super().__init__(seed, model, **params)
        if model is None:
            self.model = RandomForestRegressor
        self.model_type = "random_forest"
        self.seed = seed
        self.fitted_model = None
        self.additional_model_params: dict = params
        self.metrics = {}

    def describe(self) -> None:
        """Prints basic information about the model."""
        print(self)


class SVR_(RegressionModel):
    def __init__(self, seed=0, model=None, **params):
        super().__init__(seed, model, **params)
        if model is None:
            self.model = SVR
        self.model_type = "SVR"
        self.seed = seed
        self.fitted_model = None
        self.additional_model_params: dict = params
        self.metrics = {}

    def describe(self) -> None:
        """Prints basic information about the model."""
        print(self)


class XGBoost_(RegressionModel):
    def __init__(self, seed=0, model=None, **params):
        super().__init__(seed, model, **params)
        if model is None:
            self.model = XGBRegressor
        self.model_type = "XG_boost"
        self.seed = seed
        self.fitted_model = None
        self.additional_model_params: dict = params
        self.metrics = {}

    def describe(self) -> None:
        """Prints basic information about the model."""
        print(self)


class NeuralNet_(RegressionModel):
    def __init__(
        self,
        input_shape: np.ndarray,
        seed=0,
        model=None,
        batch_size: int = 64,
        epochs: int = 100,
        **params,
    ):
        super().__init__(seed, model, **params)
        self.model_type = "ANN"
        self.seed = seed
        self.model = model
        self.fitted_model = None
        self.additional_model_params: dict = params
        self.input_shape = input_shape
        self.metrics = {}
        self.batch_size = batch_size
        self.epochs = epochs
        self.hyperparams = None
    # early stopping?

    def __str__(self) -> None:
        """Prints basic information about the model."""
        print(f"This is a model of type: {self.model_type}.")
        if self.fitted_model is not None:
            print(
                f"It has been fit with the following structure: {self.model.summary()}"
            )
            if len(self.metrics.keys()) > 0:
                print(f"It has the following performance metrics: {self.metrics}.")

    def describe(self) -> None:
        """Prints basic information about the model."""
        print(self)

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series | pd.DataFrame):
        """Trains model based on input datasets.

        Args:
            x_train (pd.DataFrame): Feature data to be used for training.
            y_train (pd.Series | pd.DataFrame): Target data to be used for training.

        Raises:
            ValueError: If the datasets have a mismatch in number of rows.
        """
        type_assertion(x_train, pd.DataFrame | np.ndarray), type_assertion(
            y_train, pd.Series | pd.DataFrame | np.ndarray
        )
        try:
            assert x_train.shape[0] == y_train.shape[0]
        except:
            raise ValueError("x_train and y_train must have the same number of rows.")
        self.model.fit(
            x_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=False,
        )
        self.fitted_model = self.model

    def save_model(self, path: str, custom_name: str = ""):
        """Saves a model object as a keras file at the specified path.

        Args:
            path (str): Directory path where the file is to be saved.
            custom_name (str): Additional string to be added to title name.

        Raises:
            AttributeError: If the model has not yet been fit, it cannot be saved. This is to prevent saving empty models.
        """
        if self.fitted_model is None:
            raise AttributeError("The model must be fitted before it can be saved.")
        type_assertion(path, str), type_assertion(custom_name, str)
        custom_name = "_" + custom_name
        self.fitted_model.save(path + f"{self.model_type}{custom_name}.keras")

    def set_epochs(self, epochs: int) -> None:
        """Sets the model's epochs to a desired value.

        Args:
            epochs (int): Desired value of epochs.

        Raises:
            ValueError: If a negative int is passed.
        """
        type_assertion(epochs, int)
        try:
            assert epochs > 0
        except:
            raise ValueError("Value passed for epochs must be positive.")
        self.epochs = epochs

    def set_batch_size(self, batch_size: int) -> None:
        """Sets the model's batch_size to a desired value.

        Args:
            batch_size (int): Desired value for batch_size.

        Raises:
            ValueError: If a negative int is passed.
        """
        type_assertion(batch_size, int)
        try:
            assert batch_size > 0
        except:
            raise ValueError("Value passed for batch size must be positive.")
        self.batch_size = batch_size

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Inputs the dataset x into the trained model and returns the predicted targets.

        Args:
            x (pd.DataFrame): Dataset which the model will predict from. Must have the same feature set as the data which the model was originally trained on.

        Raises:
            AttributeError: If the model has not yet been fit, it cannot be used to predict.
            ValueError: If the number of features in x doesn't match the number of features in the set which was used to train the model.

        Returns:
            np.ndarray: Output of predicted targets.
        """
        try:
            assert self.fitted_model is not None
        except:
            raise AttributeError(
                "The model must be fitted before it can be used to predict."
            )
        type_assertion(x, pd.DataFrame | np.ndarray)

        return self.fitted_model.predict(x)
         # need to confirm this returns the prediction


class RegressionModelOrganizer:
    """
    Class object which can hold instances of several classification models and can perform hyperparameter optimization, training, evaluation, and prediction on sets of them.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_labels: str | List[str],
        file_directory: str = "",
        models: str | List[str] = {},
        seed: int | float = 0,
        hyperparam_grid: dict = None,
    ):
        """Constructor function for model organizer class object.

        Args:
            df (pd.DataFrame): Dataframe of preprocessed data ready to be split and fit.
            target_labels (str | List[str]): Column labels corresponding to the target variables.
            file_directory (str, optional): File directory where files will be read from and saved to by default. Defaults to "".
            models (str | List[str], optional): Dictionary containing all the models (RegressionModel objects) held by the organizer. Defaults to {}.
            seed (int | float, optional): Random seed to be used for all probablisitic seeds/random states. Defaults to 0.
            hyperparam_grid (dict, optional): Dictionary of hyperparameter suggestions to be used for hyperparameter tuning. Defaults to None.

        Raises:
            KeyError: If target labels do not correspond to column names found in df.
            IsADirectoryError: If input file_directory does not exist.
        """
        type_assertion(df, pd.DataFrame), type_assertion(seed, int | float)
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
        self.seed: int | float = seed
        np.random.seed(self.seed)
        self.x: pd.DataFrame = df.drop(target_labels, axis=1)
        self.y: pd.Series = pd.Series(df[target_labels].iloc[:, 0])
        self.features = self.x.columns
        self.n_samples: int = df.shape[0]
        self.n_features: int = self.x.shape[1]
        self.input_shape = self.x.iloc[0].shape
        try:
            assert os.path.isdir(file_directory)
        except:
            raise IsADirectoryError(
                "No directory exists at the provided path, please use an existing directory path."
            )
        self.file_directory = file_directory
        self.models = models
        self.hyperparam_grid = hyperparam_grid
        self.model_classes = {
            "lin_reg": LinearRegression_,
            "KNN": KNN_,
            "decision_tree": DecisonTree_,
            "random_forest": RandomForest_,
            "SVR": SVR_,
            "XG_boost": XGBoost_,
            "ANN": NeuralNet_,
        }
        self.x_train = None
        self.x_test = None
        self.x_val = None
        self.y_train = None
        self.y_test = None
        self.y_val = None

    def __str__(self) -> str:
        """Returns basic information about the data and models found in the organizer.

        Returns:
            str: Basic information about the loaded dataset and models.
        """
        return (
            f"Data shape: {self.x.shape} \n"
            + f"Features: {[x for x in self.x.columns]} \n"
            + f"Models: {[x.__str__() for x in self.models.keys()]}"
        )

    def create_model(self, model_type: str, **params):
        """Creates a model object corresponding to the provided model_type and adds it to the self.models dictionary.

        Args:
            model_type (str): String specifying the type of model to be created. Must be found in the following list: ["log_reg", "KNN", "decision_tree", "random_forest", "SVC", "grad_boost", "ANN"].
            name (str, optional): Optional way to return a model based on the name in the self.models dictionary. Must be passed with model_type = None. Defaults to None.

        Raises:
            KeyError: If the model_type does not match one of the model types.
        """
        type_assertion(model_type, str)
        try:
            assert model_type in self.model_classes.keys()
        except:
            raise KeyError(
                f"Value for model type must be one of {self.model_classes.keys()}"
            )
        if model_type in self.models.keys():
            while True:
                i = input(
                    f"There is already a model of this type, would you like to overwrite it? [y/n]"
                ).lower()
                if i == "n":
                    return
                if i == "y":
                    break

        if model_type == "ANN" and "input_shape" not in params.keys():
            raise ValueError(
                "For an Neural Network, value for input_shape must also be passed."
            )
        self.models[model_type] = self.model_classes[model_type](
            seed=self.seed, random_state=self.seed, **params
        )

    def load_model(self, path: str, model_type: str):
        """Loads an existing fitted model from a pickle (.pkl) file.

        Args:
            path (str): Filepath to pickled model file to load.
            model_type (str): Str key corresponding to the model type which is being loaded.

        Raises:
            FileExistsError: If the file does not exist at the given path.
            KeyError: If the str value for model_type does not match one of the acceptable models.
        """
        type_assertion(path, str), type_assertion(model_type, str)
        try:
            assert os.path.exists(os.path.join(os.getcwd(), path)) == True
            assert extension_extract(path) == "pkl" or extension_extract(path) == "keras"
        except:
            raise FileExistsError(
                "The input file path does not exist in current directory."
            )
        try:
            assert (
                model_type in available_models.keys()
            )  # does this work if this module is imported
        except:
            raise KeyError(f"The model_type must be one of {available_models.keys()}.")
        if model_type == "ANN":
            loaded_model = ks.load_model(path)
            self.create_model("ANN", input_shape=self.x.iloc[0].shape)
        else:
            with open(path, "rb") as file:
                loaded_model = pkl.load(file)
            self.create_model(model_type)
        self.models[model_type].model = loaded_model
        self.models[model_type].fitted_model = loaded_model
        self.models[model_type].n_features = self.n_features

    def get_model(self, model_type: str) -> RegressionModel:
        """Returns direct instance of a Model Class from self.models.

        Args:
            model_type (str, optional): String corresponding to the model type to be returned. Defaults to None.

        Raises:
            ValueError: If an invalid value is passed for either model_type or name or if no value are passed for either.

        Returns:
            RegressionModel: Class object of the desired model.
        """
        try:
            type_assertion(model_type, str)
            assert model_type in self.models.keys()
        except:
            raise ValueError(
                f"If model_type is passed, it must be a str matching the model_type of an existing model in the self.models dict: {self.models.keys()}."
            )
        return self.models[model_type]

    def save_models(self, models_to_save: List[str] = None, dir_path: str = None):
        """Saves model to directory for future use. Model which have not been fitted will not be saved.

        Args:
            models_to_save (List[str], optional): List of models to be saved. If None is passed, all models will be saved. Defaults to None.
            dir_path (str, optional): Directory path to save files. If none is passed, self.file_directory will be used. Defaults to None.

        Raises:
            KeyError: If any items in models_to_save do not correspond to an existing model_key.
        """
        if models_to_save is not None:
            type_assertion(models_to_save, list)
            try:
                for model_key in models_to_save:
                    assert model_key in self.models.keys()
            except:
                raise KeyError(
                    f"All the values in models_to_save must be in {self.models.keys()}"
                )
        else:
            models_to_save = self.models
        if dir_path is None:
            dir_path = self.file_directory
        for model_key in models_to_save:
            if self.models[model_key].fitted_model is not None:
                self.models[model_key].save_model(dir_path)

    def load_hyperparam_grid(
        self,
        hyperparam_grid: dict = None,
        from_json: bool = False,
        filepath: str = None,
    ):
        """Loads a hyperparameter dictionary to define the tuning seach space.

        Args:
            hyperparam_grid (dict, optional): Dictionary object. Defaults to None.
            from_json (bool, optional): Optional choice to load directly from a .json file. Defaults to False.
            filepath (str, optional): Path of .json file to use if from_json is True. Defaults to None.
        """
        if from_json is True:
            hyperparam_grid = load_JSON(filepath)
        type_assertion(hyperparam_grid, dict)
        self.hyperparam_grid = hyperparam_grid

    def run_optuna_study(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series | pd.DataFrame,
        model: RegressionModel,
        n_trials: int = 100,
        search_method: str = "base",
    ) -> tuple[int, dict]:
        """Creates and runs an Optuna study using the specified model and the hyperparameter_grid dictionary.

        Args:
            x_train (pd.DataFrame): Feature dataset to be used for training.
            y_train (pd.Series | pd.DataFrame): Target dataset to be used for training.
            model (RegressionModel): Instance of RegressionModel class which it to be optimized.
            n_trials (int, optional):  Int value which specifies the maximum number of trials the study will run for. Defaults to 100.

        Raises:
            ValueError: If value of n_trial is negative.

        Returns:
            tuple[int, dict]: Tuple containing the best_value and best_params from the Optuna study.
        """
        type_assertion(n_trials, int)
        try:
            assert n_trials > 0
        except:
            raise ValueError("The value for n_trials must be positive.")
        sampler_dict: dict = {
            "base": None,
            "grid": optuna.samplers.GridSampler,
            "random": optuna.samplers.RandomSampler,
            "gaussian": optuna.samplers.GPSampler,
        }
        type_assertion(search_method, str)
        try:
            assert search_method in sampler_dict.keys()
        except:
            raise ValueError(
                f"The value for search method must be in {sampler_dict.keys()} but {search_method} was received instead."
            )
        objective = model.__call__
        study = optuna.create_study(direction="maximize") 
        study.optimize(
            lambda trial: objective(
                trial, x=x_train, y=y_train, hyperparam_grid=self.hyperparam_grid
            ),
            n_trials=n_trials,
        )
        return study.best_params

    def run_keras_tuner(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series | pd.DataFrame,
        model: RegressionModel,
        tuner_choice: str = "hyperband",
        n_trials: int = 100,
        dir: str = None,
    ):
        """Creates and runs a keras_tuner tuning search using the specified model and hyperparameter_grid dictionary.

        Args:
            x_train (pd.DataFrame): Feature dataset to be used for training.
            y_train (pd.Series | pd.DataFrame): Target dataset to be used for training.
            model (RegressionModel): RegressionModel class obejct to be tuned.
            tuner_choice (str, optional): Tuner to be used in search. Must be from avaible options: ['random_search', 'grid_serach', 'bayesian', 'hyperband']. Defaults to 'hyperband'.
            n_trials (int, optional): Int value which specifies the maximum number of trials the study will run for search methods which allow it. Defaults to 100.
            dir (str, optional): Directory to override the self.directory for saving tuner trials. If None is passed, self.file_directory will be used. Defaults to None.

        Returns:
            Sequential, HyperParameter: Best performing model found by tuner and the corresponding HyperParameter object.
        """
        if dir is None:
            dir = self.file_directory
        clear_session(free_memory=True)
        set_seed(model.seed)
        tuners = {
            "random_search": kt.RandomSearch,
            "grid_search": kt.GridSearch,
            "bayesian": kt.BayesianOptimization,
            "hyperband": kt.Hyperband,
        }
        type_assertion(tuner_choice, str)
        try:
            assert tuner_choice in tuners.keys()
        except:
            raise KeyError(
                "Value passed for tuner_choice must be found in ['random_search', 'grid_serach', 'bayesian', 'hyperband']"
            )
        if tuner_choice == "hyperband":
            tuner = tuners[tuner_choice](
                lambda hp: compile_keras_regression_model(
                    hp,
                    input_shape=self.input_shape,
                    hyperparam_grid=self.hyperparam_grid,
                ),
                objective=kt.Objective("val_mse", direction="min"),
                max_epochs=100,
                seed=self.seed,
                directory=self.file_directory,
                project_name="ANN Hyperparameter Tuning",
            )
        else:
            tuner = tuners[tuner_choice](
                lambda hp: compile_keras_regression_model(
                    hp,
                    input_shape=self.input_shape,
                    hyperparam_grid=self.hyperparam_grid,
                ),
                objective=kt.Objective("val_mse", direction="min"),
                seed=self.seed,
                directory=self.file_directory,
                project_name="ANN Regression Hyperparameter Tuning",
                max_trials=n_trials,
            )

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                min_delta=1e-4,
                patience=5,
                verbose=0,
                mode="min",
            ),
        ]
        tuner.search(
            x_train,
            y_train,
            validation_split=0.1,
            batch_size=64,
            callbacks=callbacks,
        )
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        final_model = tuner.get_best_models(num_models=1)[0]
        return final_model, best_hps

    def split_set(
        self,
        test_size: float = 0.2,
        val_size: float = 0.0,
        shuffle: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Takes the self.df dataframe and splits it into train, test, and optionally validation sets.

        Args:
            test_size (float, optional): Determines proportion of set to be allocated to the test set. Must be in range[0,1). Defaults to 0.2.
            val_size (float, optional): Determines proportion of set to be allocated to the test set. Must be in range[0,1). Defaults to 0.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: train, test, and validation DataFrames. Note, that if validation is set to zero, only the train and test frames will be returned.
        """
        x = self.x
        y = self.y
        type_assertion(test_size, float), type_assertion(val_size, float)
        try:
            assert (
                test_size >= 0
                and test_size < 1
                and val_size >= 0
                and val_size < 1
                and test_size + val_size <= 1
            )
        except:
            raise ValueError(
                "The values for test_size and val_size must be floats in the range [0,1), and must not have a sum exceeding 1."
            )
        if val_size == 0:
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=test_size, random_state=self.seed, shuffle=shuffle
            )
            self.x_train = x_train
            self.x_test = x_test
            self.y_train = y_train
            self.y_test = y_test
            return x_train, x_test, y_train, y_test
        else:
            split_size1 = test_size + val_size
            split_size2 = val_size / (val_size + test_size)
            x_train, intermediary_df_x, y_train, intermediary_df_y = train_test_split(
                x, y, test_size=split_size1, random_state=self.seed
            )
            x_test, x_val, y_test, y_val = train_test_split(
                intermediary_df_x,
                intermediary_df_y,
                test_size=split_size2,
                random_state=self.seed,
                shuffle=shuffle,
            )
            self.x_train = x_train
            self.x_test = x_test
            self.x_val = x_val
            self.y_train = y_train
            self.y_test = y_test
            self.y_val = y_val
            return x_train, x_test, x_val, y_train, y_test, y_val

    def predict_set(
        self,
        x: pd.DataFrame,
        models_to_use: list = None,
    ) -> Dict[str, np.ndarray]:
        """Returns dictionary of predictions of a set of models.

        Args:
            x (pd.DataFrame): Feature dataset to be used for prediction.
            models_to_use (list, optional): List to specify which models are to be used for prediction. If None is passed, all fitted models will be used. Defaults to None.

        Raises:
            KeyError: If any items in models_to_use do not correspond to an existing model_key.

        Returns:
            Dict[str, np.ndarray]: Dictionary consisting of model_keys and the corresponding model's prediction set.
        """
        if models_to_use is not None:
            type_assertion(models_to_use, list)
            try:
                for model_key in models_to_use:
                    assert model_key in self.models.keys()
            except:
                raise KeyError(
                    f"All the values in models_to_use must be in {self.models.keys()}"
                )
        else:
            models_to_use = list(self.models.keys())
        y_preds: Dict[str, np.ndarray] = {}
        for model_key in models_to_use:
            if self.models[model_key].fitted_model is not None:
                y_preds[model_key] = self.models[model_key].predict(x)
        return y_preds

    def score_set(
        self,
        x: pd.DataFrame = None,
        y_true: pd.Series | pd.DataFrame = None,
        models_to_score: list = None,
        y_preds: Dict[str, np.ndarray] = None,
        overwrite_model_metrics: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Returns dictionary of scoring metrics of a set of models.

        Args:
            x (pd.DataFrame, optional): Feature dataset to optionally override self.x_test if passed. Defaults to None.
            y_true (pd.Series | pd.DataFrame, optional): Target dataset to optionally override self.y_test if passed. Defaults to None.
            models_to_score (list, optional):  List to specify which models are to be used for scoring. If None is passed, all models will be used. Defaults to None.
            y_preds (Dict[str, np.ndarray], optional): Target predictions dataset to optionally skip re-predicting. Defaults to None.
            overwrite_model_metrics (bool, optional): Determines if model metrics are overwritten or not in the RegressionModel class object. Defaults to True.

        Raises:
            KeyError: If any items in models_to_score do not correspond to an existing model_key.

        Returns:
            Dict[str, np.ndarray]: Dictionary consisting of model_keys and the corresponding model's scoring metrics.
        """
        if models_to_score is not None:
            type_assertion(models_to_score, list)
            try:
                for model_key in models_to_score:
                    assert model_key in self.models.keys()
            except:
                raise KeyError(
                    f"All the values in models_to_fit must be in {self.models.keys()}"
                )
        else:
            models_to_score = list(self.models.keys())
        if x is None:
            x = self.x_test
        if y_true is None:
            y_true = self.y_test
        scores: Dict = {}
        if y_preds is None:
            y_preds: Dict[str, np.ndarray] = self.predict_set(
                x, models_to_use=models_to_score
            )
        for model_key in models_to_score:
            scores[model_key] = self.models[model_key].score(y_true, y_preds[model_key])
            if overwrite_model_metrics is True:
                self.models[model_key].metrics = scores[model_key]
        return scores

    def fit_set(
        self,
        models_to_fit: list = None,
        get_scores: bool = False,
        retrain_on_full_set: bool = False,
        param_tuning: bool = True,
        hyper_params: dict = None,
        n_trials: int = 100,
        test_size: float = 0.2,
        # val_size: float = 0.0, # not yet implemented
        optuna_search_method: str = "base",
        keras_tuner_search_method: str = "hyperband",
        **params,
    ) -> None:
        """Fits a set of models

        Args:
            models_to_fit (list, optional): List of model keys to be fit. If None is passed, all existing models will be used. Defaults to None.
            get_scores (bool, optional): If the models are scored after fitting. Defaults to False.
            retrain_on_full_set (bool, optional): If the models should be refit on the entire feature dataset. Warning: this could lead to data leaking issues if models are scores on data they trained on. Defaults to False.
            param_tuning (bool, optional): If the models are to be hyperparameter tuned. Defaults to True.
            n_trials (int, optional): Maximum number of trials to be run in hyperparameter tuning. Defaults to 100.
            hyper_params (dict, optional): Dictionary of hyperparamers to be passed to the models if hyperparameter tuning is not performed. Defaults to None.
            test_size (float, optional): Portion of the dataset to be split into the testing set, must be in range (0,1). Defaults to 0.2.
            optuna_search_method (str, optional): Method to be used for tuning with optuna, possible options are: ["base", "grid", "random", "gaussian"]. Defaults to "base".
            keras_tuner_search_method (str, optional): Method to be used for tuning with keras-tuner, possible options are ['random_search', 'grid_serach', 'bayesian', 'hyperband']. Defaults to "hyperband".

        Raises:
            ValueError: If test_size or n_trials is an unacceptable value.
            KeyError: If the keys passed in modes_to_fit or in the hyper_params dict do not correspond to an existing model class object.
        """
        type_assertion(get_scores, bool), type_assertion(
            retrain_on_full_set, bool
        ), type_assertion(param_tuning, bool),
        type_assertion(test_size, float), type_assertion(n_trials, int)
        if test_size <= 0 or test_size >= 1:
            raise ValueError("The value for test_size must be in the range (0,1).")
        if n_trials <= 0:
            raise ValueError("The value for n_trials must be strictly positive.")
        x_train, x_test, y_train, y_test = self.split_set(
            test_size=test_size,
            val_size=0.0,  # val_size,
        )
        model_params: dict = {}
        if models_to_fit is not None:
            type_assertion(models_to_fit, list)
            try:
                for model_key in models_to_fit:
                    assert model_key in self.models.keys()
            except:
                raise KeyError(
                    f"All the values in models_to_fit must be in {self.models.keys()}"
                )
        else:
            models_to_fit = list(self.models.keys())
        if param_tuning is True:
            type_assertion(optuna_search_method, str), type_assertion(
                keras_tuner_search_method, str
            )
            for model_key in models_to_fit:
                if model_key != "ANN":
                    model_params[model_key] = self.run_optuna_study(
                        x_train=x_train,
                        y_train=y_train,
                        model=self.models[model_key],
                        n_trials=n_trials,
                        search_method=optuna_search_method,
                    )
                    self.models[model_key].model = available_models[model_key](
                        **model_params[model_key]
                    )
                else:
                    self.models[model_key].model, model_params[model_key] = (
                        self.run_keras_tuner(
                            x_train=x_train,
                            y_train=y_train,
                            model=self.models[model_key],
                            n_trials=n_trials,
                            tuner_choice=keras_tuner_search_method,
                            **params,
                        )
                    )
                    self.models[model_key].model_params = model_params[model_key]
        else:
            for model_key in models_to_fit:
                try:
                    assert model_key in hyper_params.keys()
                except:
                    raise KeyError(
                        "Hyperparams must be passed for all models in models_to_fit."
                    )
                model_params[model_key] = hyper_params[model_key]
                if model_key != "ANN":
                    self.models[model_key] = available_models[model_key](
                        **hyper_params[model_key]
                    )
                    model_class = self.models[model_key]
                    model_class.fit(self.x, self.y)
                    self.models[model_key].hyperparams = model_params[model_key]
                else:
                    self.models[model_key] = compile_keras_regression_model_manual(
                        **hyper_params[model_key]
                    )
                    model_class = self.models[model_key]
                    model_class.fit(self.x, self.y)
                    self.models[model_key].hyperparams = model_params[model_key]
        if get_scores is True:
            self.score_set(x_test, y_test, models_to_score=models_to_fit)
        if retrain_on_full_set is True:
            for model_key in models_to_fit:
                model_class = self.models[model_key]
                model_class.fit(self.x, self.y)
                self.models[model_key].hyperparams = model_params[model_key]
        else:
            for model_key in models_to_fit:
                model_class = self.models[model_key]
                model_class.fit(x_train, y_train)
                self.models[model_key].hyperparams = model_params[model_key]

    def compare_models(
        self,
        models_to_compare: List[str] = None,
        plot: bool = False,
        display_plot: bool = True,
        save_plot: bool = False,
        save_dir: str = "",
        save_name: str = "regression_metrics",
    ) -> pd.DataFrame:
        """Compares scoring metrics on a set of fitted models with the option to plot a series of barcharts comparing these metrics. Models must already be fit and scored.

        Args:
            models_to_compare (List[str], optional): List of model_keys corresponding to the models to be compared. If none, all applicable models will be compared. Defaults to None.
            plot (bool, optional): Specifies if the model comparison plot should be created. Defaults to False.
            display_plot (bool, optional): Specifies if the plot will be displayed or not. Defaults to True.
            save_plot (bool, optional): Specifies if the plot should be saved if created. Defaults to False.
            save_dir (str, optional): Directory where the plot is to be saved. Defaults to "".
            save_name (str, optional): Filename the plot will be saved as. Defaults to "regression_metrics".

        Raises:
            KeyError: If any items in models_to_compare do not correspond to an existing model_key.
            ValueError: If the save_dir does not correspond to an existing directory.

        Returns:
            pd.DataFrame: DataFrame of all models compared and the metrics they received.
        """
        type_assertion(plot, bool), type_assertion(save_plot, bool), type_assertion(
            save_dir, str
        ), type_assertion(save_name, str)
        if models_to_compare is not None:
            type_assertion(models_to_compare, list)
            try:
                for model_key in models_to_compare:
                    assert (
                        model_key in self.models.keys()
                        and self.models[model_key].fitted_model is not None
                    )
            except:
                raise KeyError(
                    f"All the values in models_to_compare must be in {self.models.keys()}"
                )
        else:
            models_to_compare = [
                model
                for model in self.models.keys()
                if self.models[model].fitted_model is not None
            ]
        metrics = {}
        for model_key in models_to_compare:
            try:
                assert "mean_squared_error" in self.models[model_key].metrics.keys()
            except:
                raise KeyError("The models must already be scored to compare them.")
            metrics[model_key] = {
                "Mean Squared Error": self.models[model_key].metrics["mean_squared_error"],
                "Mean Absolute Error": self.models[model_key].metrics["mean_absolute_error"],
                "R2 Score": self.models[model_key].metrics["r2_score"],
                "Root Mean Squared Error": self.models[model_key].metrics["root_mean_squared_error"],
            }
        metrics_df = pd.DataFrame(metrics).T
        if plot is True:
            plot_num = 1
            fig, axs = plt.subplots(2, 2)
            fig.tight_layout()
            fig.set_size_inches(14, 10)
            for metric in metrics_df.columns:
                plt.subplot(2, 2, plot_num)
                bars = plt.bar(x=metrics_df.index, height=metrics_df[metric])
                plt.bar_label(bars, fmt="%.2f", label_type="center", color="w")
                plt.title(f"{metric}")
                plt.xticks(rotation=25)
                plot_num += 1
            if save_plot is True:
                try:
                    assert os.path.isdir(save_dir)
                except:
                    raise ValueError(
                        "The provided directory for save_dir, does not exist. Please use a valid directory."
                    )
                plt.savefig(save_dir + save_name + ".png")
            if display_plot is True:
                plt.show()
        return metrics_df
