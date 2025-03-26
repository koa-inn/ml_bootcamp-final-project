import os
import numpy as np
import pandas as pd
from typing import List, Dict
import optuna
import pickle as pkl
import random
from abc import abstractmethod

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session
import tensorflow.keras.saving as ks
import keras_tuner as kt


from utils.utils import type_assertion, save_to_JSON, load_JSON, extension_extract


available_models = {
    "log_reg": LogisticRegression,
    "KNN": KNeighborsClassifier,
    "tree": DecisionTreeClassifier,
    "rand_for": RandomForestClassifier,
    "SVC": SVC,
    "grad_boost": GradientBoostingClassifier,
    "ANN": "ANN",
}

metric_fns = {
    "f1": f1_score,
    "acc": accuracy_score,
    "rec": recall_score,
    "prec": precision_score,
}

hyperparam_grid = load_JSON('/Users/koa/Documents/ML Course/capstone/bootcamp-ml-framework/hyperparams.json')['classification']


def create_optuna_suggestion(
    trial, model_type: str, parameter: str, hyperparam_grid
):  # may need to add a grid dictionary arg, considering moving to utils
    """
    Reads from hyperparameter_grid dictionary and returns the specified parameter as a optuna trial suggestion.
    """
    parameter_dict = hyperparam_grid[model_type][parameter]
    if parameter_dict["type"] == "int":
        return trial.suggest_int(
            parameter,
            parameter_dict["low"],
            parameter_dict["high"],
            step=parameter_dict["step"],
            log=parameter_dict["log"],
        )
    if parameter_dict["type"] == "float":
        return trial.suggest_float(
            parameter,
            parameter_dict["low"],
            parameter_dict["high"],
            step=parameter_dict["step"],
            log=parameter_dict["log"],
        )
    if parameter_dict["type"] == "uniform":
        return trial.suggest_uniform(
            parameter,
            parameter_dict["low"],
            parameter_dict["high"],
            step=parameter_dict["step"],
        )
    if parameter_dict["type"] == "categorical":
        return trial.suggest_categorical(parameter, parameter_dict["choices"])


def create_param_dict(trial, model_type: str, hyperparam_grid) -> dict:
    """
    Creates a dictionary of optuna suggestions to be used by an Optuna objective.
    """
    params: dict = {}
    model_dict: dict = hyperparam_grid[model_type]
    for key in model_dict:
        params[key] = create_optuna_suggestion(
            trial, model_type, key, hyperparam_grid=hyperparam_grid
        )
    return params


def create_kt_suggestion(hp, model_type, parameter, param_dict=hyperparam_grid):

    parameter_dict = param_dict[model_type][parameter]
    if parameter_dict["type"] == "int":
        return hp.Int(
            parameter,
            min_value=parameter_dict["low"],
            max_value=parameter_dict["high"],
            step=parameter_dict["step"],
            sampling=parameter_dict["sampling"],
        )
    if parameter_dict["type"] == "float":
        return hp.Float(
            parameter,
            min_value=parameter_dict["low"],
            max_value=parameter_dict["high"],
            step=parameter_dict["step"],
            sampling=parameter_dict["sampling"],
        )
    if parameter_dict["type"] == "categorical":
        return hp.Choice(parameter, values=parameter_dict["choices"])
    if parameter_dict["type"] == "bool":
        return hp.Boolean(parameter)


def compile_keras_model(
    hp, input_shape: np.ndarray, n_classes: int
):  # Builds classifier ann ## Need to figure out how get input shape and output shape
    """Compiles a keras model to be pased to a keras_tuner tuner for hyperparameter tuning.

    Args:
        hp (_type_): _description_
        input_shape (np.ndarray): _description_
        n_classes (_type_): _description_
    """
    type_assertion(n_classes, int)

    model = Sequential()

    n_dense_layers = create_kt_suggestion(
        hp, model_type="ANN", parameter="n_dense_layers"
    )
    dense_units = create_kt_suggestion(hp, model_type="ANN", parameter="dense_units")
    dropout_rate = create_kt_suggestion(hp, model_type="ANN", parameter="dropout_rate")
    learning_rate = create_kt_suggestion(
        hp, model_type="ANN", parameter="learning_rate"
    )
    # ^ is there a more generalizable way to do this?

    optimizer = Adam(learning_rate=learning_rate)

    for i in range(0, n_dense_layers):
        model.add(Dense(units=dense_units, activation="relu"))
        model.add(Dropout(rate=dropout_rate))

    if n_classes == 2:
        model.add(Dense(units=1, actvation="sigmoid"))
        model.compile(
            optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
        )
    else:
        model.add(Dense(units=n_classes, activation="softmax"))
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    return model

def compile_keras_model_manual(n_layers: int, dense_units: int, dropout_rate: float, learning_rate: float, n_classes: int, activation: str = 'relu'):
    type_assertion(n_layers, int),type_assertion(dense_units, int),type_assertion(dropout_rate, float|int),type_assertion(learning_rate, float|int), type_assertion(n_classes, int) 
    try:
        assert n_layers > 0 and dense_units > 0 and dropout_rate > 0 and learning_rate > 0
    except:
        raise ValueError("The values for n_layers, dense_units, dropout_rate, and learning_rate must be strictly positive.")
    try:
        assert n_classes > 1
    except:
        raise ValueError()
    try:
        assert activation in ['relu', 'leaky_relu', 'tanh', 'sigmoid', 'softmax']
    except:
        raise ValueError("The value for activation must be one of: ['relu', 'leaky_relu', 'tanh', 'sigmoid', 'softmax']")
    
    model = Sequential
    optimizer = Adam(learning_rate=learning_rate)

    for layer in n_layers:
        model.add(Dense(units = dense_units, activation=activation))
        model.add(Dropout(rate=dropout_rate))

    if n_classes == 2:
        model.add(Dense(units=1, actvation="sigmoid"))
        model.compile(
            optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
        )
    else:
        model.add(Dense(units=n_classes, activation="softmax"))
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    return model        

class ClassificationModel:
    """
    Class for generic classification model.
    Properties:
    Methods:
    """

    def __init__(self, seed: int | float = 0, model: object = None, **params):
        """ """
        self.model = model
        self.model_type: str = None
        self.seed: int | float = seed
        self.additional_model_params: dict = params
        self.fitted_model = None
        self.metrics = {}
        self.hyperparams = None

    def __call__(
        self,
        trial,
        x,
        y,
        metric_choice: str = "f1",
        non_binary_averaging="weighted",
        **params,
    ):
        """
        Call method for running Optuna trials
        """
        if self.model_type != "ANN":
            try:
                assert metric_choice in ["f1", "accuracy", "precision", "recall"]
                assert non_binary_averaging in ["micro", "macro", "weighted"]
            except:
                raise ValueError(
                    "The value for metric_choice must be in ['f1', 'accuracy', 'precision', 'recall'] and the value for non_binary_averaging must be in ['micro', 'macro', 'weighted']"
                )
            params: dict = create_param_dict(trial=trial, model_type=self.model_type, hyperparam_grid=hyperparam_grid)
            model = self.model(**params)
            model.fit(x, y)
            n_classes = y.nunique()
            if metric_choice != "accuracy":
                if n_classes != 2:
                    metric_choice = metric_choice + "_" + non_binary_averaging
            metric = cross_val_score(
                model, x, y, scoring=metric_choice, cv=5
            )  #: float = metric_fns[metric_to_use](y, y_pred, average='macro')
            metric = np.mean(metric)
            return metric

    def save_model(self, path: str):
        """
        Saves a model object as a pickle file at the specified path.
        """
        try:
            assert self.fitted_model is not None
        except:
            raise AttributeError("The model must be fitted before it can be saved.")
        pkl.dump(self.fitted_model, f"{path}{self.model_type}.pkl")

    def fit(self, x_train, y_train):
        """
        Trains model based on input parameters.
        """
        self.model.fit(x_train, y_train)
        self.hyperparams = self.model.get_params()
        self.fitted_model = self.model

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """ """
        try:
            assert self.fitted_model is not None
        except:
            raise AttributeError(
                "The model must be fitted before it can be used to predict."
            )
        type_assertion(x, pd.DataFrame)
        return self.fitted_model.predict(x)

    def score(
        self,
        y_true: pd.Series | pd.DataFrame,
        y_pred: pd.Series | pd.DataFrame = None,
        x: pd.DataFrame = None,
        average_for_nonbinary: str = "weighted" 
    ) -> Dict[str, float]:
        """Returns performance metrics (f1 score, accuracy, recall, precision) for the model.

        Args:
            y_true (pd.Series | pd.DataFrame): True target values to be passed and scored against.
            y_pred (pd.Series | pd.DataFrame, optional): _description_. Defaults to None.
            x (pd.DataFrame, optional): DataFrame to be scores upon in comparison to y_true. If passed, value for y_pred will be overwritten and a new fitting will take places using x. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            Dict[str, float]: _description_
        """
        if y_pred is None:
            try:
                assert x is not None
            except:
                raise ValueError("A value must be passed for either y_pred or x")
        n_classes = y_true.nunique()
        if n_classes == 2:
            average = 'binary'
        else:
            type_assertion(average_for_nonbinary, str)
            try: 
                assert average_for_nonbinary in ['weighted', 'micro', 'macro']
            except:
                raise ValueError(f"The value for average_for_nonbinary must be found in ['weighted', 'micro', 'macro'], but {average_for_nonbinary} was receieved")
            average = average_for_nonbinary
        if x is not None:
            y_pred = self.predict(x)
        output = {}
        output["f1"] = f1_score(y_true, y_pred, average=average)
        output["accuracy"] = accuracy_score(y_true, y_pred)
        output["recall"] = recall_score(y_true, y_pred, average=average)
        output["precision"] = precision_score(y_true, y_pred, average=average)
        return output


class LogisticRegression_(ClassificationModel):
    """ """

    # need to include multiclass specification?
    def __init__(self, seed=0, model=None, **params):
        super().__init__(seed, model, **params)
        if model is None:
            self.model = LogisticRegression
        self.model_type: str = "log_reg"
        self.seed = seed
        self.fitted_model = None
        self.additional_model_params: dict = params
        self.metrics = {}


class KNN_(ClassificationModel):
    def __init__(self, seed=0, model=None, **params):
        super().__init__(seed, model, **params)
        if model is None:
            self.model = KNeighborsClassifier
        self.model_type = "KNN"
        self.seed = seed
        self.fitted_model = None
        self.additional_model_params: dict = params
        self.metrics = {}


class DecisonTree_(ClassificationModel):
    def __init__(self, seed=0, model=None, **params):
        super().__init__(seed, model, **params)
        if model is None:
            self.model = DecisionTreeClassifier
        self.model_type = "tree"
        self.seed = seed
        self.fitted_model = None
        self.additional_model_params: dict = params
        self.metrics = {}


class RandomForest_(ClassificationModel):
    def __init__(self, seed=0, model=None, **params):
        super().__init__(seed, model, **params)
        if model is None:
            self.model = RandomForestClassifier
        self.model_type = "rand_for"
        self.seed = seed
        self.fitted_model = None
        self.additional_model_params: dict = params
        self.metrics = {}


class SVC_(ClassificationModel):
    def __init__(self, seed=0, model=None, **params):
        super().__init__(seed, model, **params)
        if model is None:
            self.model = SVC
        self.model_type = "SVC"
        self.seed = seed
        self.fitted_model = None
        self.additional_model_params: dict = params
        self.metrics = {}


class GradBoost_(ClassificationModel):
    def __init__(self, seed=0, model=None, **params):
        super().__init__(seed, model, **params)
        if model is None:
            self.model = GradientBoostingClassifier
        self.model_type = "grad_boost"
        self.seed = seed
        self.fitted_model = None
        self.additional_model_params: dict = params
        self.metrics = {}


class NeuralNet_(ClassificationModel):
    def __init__(
        self, input_shape: np.ndarray, n_classes: int, seed=0, model=None, batch_size: int = 64, epochs: int = 100, **params
    ):
        super().__init__(seed, model, **params)
        self.model_type = "ANN"
        self.seed = seed
        self.model = model
        self.fitted_model = None
        self.additional_model_params: dict = params
        self.n_classes: int = n_classes
        self.input_shape = input_shape
        self.metrics = {}
        self.batch_size = batch_size
        self.epochs = epochs
        self.hyperparams = None

        # early stopping, batch size, n_epochs

    def fit(self, x_train, y_train):
        """ """
        self.model.fit(
            x_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=False,
        )
        self.fitted_model = self.model

    def save_model(self, path: str):
        if self.fitted_model is None:
            raise AttributeError(
                "The model must be fitted before it can be used to predict."
            )
        type_assertion(path, str)
        self.fitted_model.save(path)

    def set_epochs(self, epochs: int) -> None:
        type_assertion(epochs, int)
        try:
            assert epochs > 0
        except:
            raise ValueError("Value passed for epochs must be positive.")
        self.epochs = epochs

    def get_weights(self):
        if self.fitted_model is None:
            raise AttributeError(
                "The model must be fitted before it can be used to predict."
            )

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """ """
        try:
            assert self.fitted_model is not None
        except:
            raise AttributeError(
                "The model must be fitted before it can be used to predict."
            )
        type_assertion(x, pd.DataFrame)
        if self.n_classes > 2:
            return np.argmax(self.fitted_model.predict(x), axis=1)
        else:
            return np.round(self.fitted_model.predict(x))



class ClassificationModelOrganizer:
    """
    Class which can hold instances of several classification models and can perform hyperparameter optimization, model training and evaluation, and prediction on sets of them.
    Properties:
    Methods:
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
        self.x: pd.DataFrame = df.drop(target_labels, axis=1)
        self.y: pd.Series = pd.Series(df[target_labels].iloc[:, 0])
        self.n_classes: int = self.y.nunique()
        if self.n_classes == 1:
            raise ValueError(
                "Classification dataset has only one class, at least two are required to meaningful classfication."
            )
        self.n_samples: int = df.shape[0]
        self.n_features: int = self.x.shape[1]
        self.input_shape = self.x.iloc[0].shape
        try:
            assert os.path.isdir(file_directory)
        except:
            raise IsADirectoryError("No directory exists at the provided path, please use an existing directory path.")
        self.file_directory = file_directory
        self.models = models
        self.hyperparam_grid = hyperparam_grid
        self.model_classes = {
            "log_reg": LogisticRegression_,
            "KNN": KNN_,
            "tree": DecisonTree_,
            "rand_for": RandomForest_,
            "SVC": SVC_,
            "grad_boost": GradBoost_,
            "ANN": NeuralNet_,
        }
        self.x_train = None
        self.x_test = None
        self.x_val = None
        self.y_train = None
        self.y_test = None
        self.y_val = None
        self.tuning_seach_method = 'default' #add in 

    def create_model(self, model_type: str, **params):
        """Creates a model object corresponding to the provided model_type and adds it to the self.models dictionary.

        Args:
            model_type (str): String specifying the type of model to be created. Must be found in the following list: ["log_reg", "KNN", "tree", "rand_for", "SVC", "grad_boost", "ANN"].
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
                if i == "y":
                    break

        if model_type == "ANN" and (
            "n_classes" not in params.keys() or "input_shape" not in params.keys()
        ):
            raise ValueError(
                "For an Neural Network, values for input_shape and epochs must also be passed."
            )
        self.models[model_type] = self.model_classes[model_type](
            seed=self.seed, **params
        )

    def load_model(self, path: str, model_type: str):
        """Loads an existing model from a pickle (.pkl) file.

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
            assert extension_extract(path) == "plk"
        except:
            raise FileExistsError(
                "The input file path does not exist in current directory."
            )
        try:
            assert (
                model_type in available_models.keys
            )  # does this work if this module is imported
        except:
            raise KeyError(f"The model_type must be one of {available_models.keys}.")
        if model_type == "ANN":
            loaded_model = ks.load_model(path)
            pass
        else:
            loaded_model = pkl.load(path)
            self.models[model_type] = loaded_model

    def get_model(self, model_type: str) -> ClassificationModel:
        """Returns direct instance of a Model Class from self.models.

        Args:
            model_type (str, optional): String corresponding to the model type to be returned. The first model of that model_type found will be returned. Defaults to None.

        Raises:
            ValueError: If an invalid value is passed for either model_type or name or if no value are passed for either.

        Returns:
            ClassificationModel: Class object of the desired model.
        """
        try:
            type_assertion(model_type, str)
            assert model_type in self.models.keys()
        except:
            raise ValueError(
                "If model_type is passed, it must be a str matching the model_type of an existing model in the self.models dict."
            )
        return self.models[model_type]

    def save_models(self, models_to_save: List[str] = None, dir_path: str = None):
        pass

    def load_hyperparam_grid(
        self,
        hyperparam_grid: dict = None,
        from_json: bool = False,
        filepath: str = None,
    ):
        if from_json is True:
            hyperparam_grid = load_JSON(filepath)
        type_assertion(hyperparam_grid, dict)
        self.hyperparam_grid = hyperparam_grid

    # not really sure where the following 2 functions should exist? here or in the model classes or globaly. Not really sure how to make globaly work though.
    def run_optuna_study(
        self, x_train, y_train, model: ClassificationModel, n_trials: int = 100
    ) -> tuple[int, dict]:
        """Creates and runs an Optuna study using the specified model and the hyperparameter_grid dictionary.

        Args:
            x_train ():
            y_train ():
            model (ClassificationModel): Instance of ClassficationModel class which it to be optimized.
            n_trials (int, optional):  Int value which specifies the maximum number of trials the study will run for.. Defaults to 100.

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
        objective = model.__call__
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(trial, x=x_train, y=y_train), n_trials=n_trials
        )
        return study.best_params

    def run_keras_tuner(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series | pd.DataFrame,
        model: ClassificationModel,
        tuner_choice: str = "hyperband",
        n_trials: int = 100,
        dir: str = "",
    ):
        """Creates and runs a keras_tuner tuning search using the specified model and hyperparameter_grid dictionary.

        Args:
            x_train (pd.DataFrame): Training data feature set.
            y_train (pd.Series | pd.DataFrame): Training data target set.
            model (ClassificationModel): ClassificationModel class obejct to be tuned.
            tuner_choice (str, optional): Tuner to be used in search. Must be from avaible options: ['random_search', 'grid_serach', 'bayesian', 'hyperband']. Defaults to 'hyperband'.
            n_trials (int, optional): _description_. Defaults to 100.
            dir (str, optional): _description_. Defaults to "".

        Returns:
            _type_: _description_
        """
        clear_session(free_memory=True)

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
        tuner = tuners[tuner_choice](
            lambda hp: compile_keras_model(
                hp, input_shape=self.input_shape, n_classes=self.n_classes
            ),
            objective="val_accuracy",
            max_epochs=100,
            seed=self.seed,
            directory = self.file_directory,
            project_name = "ANN Hyperparameter Tuning"
        )
        callbacks = [
            EarlyStopping(
                monitor="val loss",
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
        self, test_size: float = 0.2, val_size: float = 0.0, shuffle: bool = True,
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
        y_true: pd.Series | pd.DataFrame,
        models_to_use: list = None,
    ) -> pd.DataFrame:
        # add option to pass parameters or do tuning
        if models_to_use is not None:
            type_assertion(models_to_use, list)
            try:
                for model_key in models_to_use:
                    assert model_key in self.models.keys()
            except:
                raise KeyError(
                    f"All the values in models_to_fit must be in {self.models.keys()}"
                )
        else:
            models_to_use = list(self.models.keys())
        y_preds: Dict[str, np.ndarray] = {}
        for model_key in models_to_use:
            y_preds[model_key] = self.models[model_key].predict(x)
        return y_preds

    def score_set(
        self,
        x: pd.DataFrame = None,
        y_true: pd.Series | pd.DataFrame = None,
        models_to_score: list = None,
        y_preds: Dict[str, np.ndarray] = None,
        overwrite_model_metrics: bool = True,
    ) -> pd.DataFrame:

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
                x, y_true, models_to_use=models_to_score
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
        test_size: float = 0.2,
        val_size: float = 0.0,
        **params
    ):
        x_train, x_test, y_train, y_test = self.split_set(
            test_size=test_size, val_size=val_size,
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
        for model_key in models_to_fit:
            if model_key != "ANN":
                model_params[model_key] = self.run_optuna_study(
                    x_train=x_train,
                    y_train=y_train,
                    model=self.models[model_key],
                    n_trials=100,
                )
                self.models[model_key].model = available_models[model_key](
                    **model_params[model_key]
                )
            else:
                self.models[model_key].model, model_params[model_key] = (
                    self.run_keras_tuner(
                        x_train=x_train, y_train=y_train, model=self.models, **params,
                    )
                )
                self.models[model_key].model_params = model_params[model_key]
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
        
 