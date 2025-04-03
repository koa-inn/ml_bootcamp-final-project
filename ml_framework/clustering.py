import os
import numpy as np
import pandas as pd
from typing import List, Dict
import optuna
import pickle as pkl
import random
from abc import ABC, abstractmethod
import math


from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt


from utils.utils import (
    type_assertion,
    save_to_JSON,
    load_JSON,
    extension_extract,
    create_optuna_suggestion,
    create_param_dict,
)

available_models = {
    "k_means": KMeans,
    "agglomerative": AgglomerativeClustering,
    "mean_shift": MeanShift,
    "dbscan": DBSCAN,
}

metric_fns = {
    "silhouette_score": silhouette_score, # maximize
    "davies_bouldin_score": davies_bouldin_score, # minimize
}



class ClusteringModel(ABC):
    """
    Abstract Class for generic clustering model.
    """

    def __init__(self, seed: int | float = 0, model: object = None, **params):
        """Contructor basis for all Clustering models.

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
        hyperparam_grid: dict,
        metric_choice: str = "silhouette_score",
        **params,
    ) -> float:
        """Method for use only by optuna trials to choose hyperparams from grid, fit model, and return metric.

        Args:
            trial: Optuna trial
            x (pd.DataFrame): Feature dataset.
            hyperparam_grid (dict): Hyperparameter dictionary to specify gird search space.
            metric_choice (str, optional): Metric to optimize hyperparameters for from: ["silhouette_score", "davies_bouldin_score"]. Defaults to "f1".

        Raises:
            ValueError: If a non-valid str is passed for metric_choice.

        Returns:
            float: Specified metric which was achieved by the model in this trial.
        """
        try:
            assert metric_choice in ["silhouette_score", "davies_bouldin_score"]
        except:
            raise ValueError(
                "The value for metric_choice must be in ['silhouette_score', 'davies_bouldin_score']."
            )
        params: dict = create_param_dict(
            trial=trial, model_type=self.model_type, hyperparam_grid=hyperparam_grid
        )
        model = self.model(**params)
        labs = model.fit_predict(x)
        unique, counts = np.unique(labs, return_counts=True)
        k = len(unique)
        if k == 1:
            metric = -np.inf
        if min(counts) < 50:
            metric = -np.inf
        else:
            metric = metric_fns[metric_choice](x, labs)
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

    def fit(self, x: pd.DataFrame):
        """Trains model based on input data.

        Args:
            x (pd.DataFrame): Feature data to be used for clustering.
        """
        type_assertion(x, pd.DataFrame | np.ndarray)
        self.model.fit(x)
        self.hyperparams = self.model.get_params()
        self.fitted_model = self.model
        self.n_features = x.shape[1]


    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Inputs the dataset x into the trained model and returns the assigned cluster labels.

        Args:
            x (pd.DataFrame): Dataset which the model will predict from. Must have the same feature set as the data which the model was originally trained on.

        Raises:
            AttributeError: If the model has not yet been fit, it cannot be used to predict.
            ValueError: If the number of features in x doesn't match the number of features in the set which was used to train the model.

        Returns:
            np.ndarray: Output of cluster labels.
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
        return self.fitted_model.fit_predict(x)

    def score(
        self,
        x: pd.DataFrame,
        cluster_labels: np.ndarray | pd.Series | pd.DataFrame = None,
    ) -> Dict[str, float]:

        if cluster_labels is None:
            cluster_labels = self.predict(x)
        output = {}
        output["silhouette_score"] = silhouette_score(x, cluster_labels, random_state=self.seed)
        output["davies_bouldin_score"] = davies_bouldin_score(x, cluster_labels)
        self.metrics = output
        return output


class KMeans_(ClusteringModel):
    def __init__(self, seed = 0, model = None, **params):
        super().__init__(seed, model, **params)
        self.seed: int | float = seed
        self.model = KMeans
        self.model_type: str = "k_means"
        self.additional_model_params = params
        self.metrics = {}
        self.fitted_model = None
        self.hyperparams = None
        self.n_features = None

    def describe(self) -> None:
        """Prints basic information about the model."""
        print(self)

class MeanShift_(ClusteringModel):
    def __init__(self, seed = 0, model = None, **params):
        super().__init__(seed, model, **params)
        self.seed: int | float = seed
        self.model = MeanShift
        self.model_type: str = "mean_shift"
        self.additional_model_params = params
        self.metrics = {}
        self.fitted_model = None
        self.hyperparams = None
        self.n_features = None

    def describe(self) -> None:
        """Prints basic information about the model."""
        print(self)

class Agglomerative_(ClusteringModel):
    def __init__(self, seed = 0, model = None, **params):
        super().__init__(seed, model, **params)
        self.seed: int | float = seed
        self.model = AgglomerativeClustering
        self.model_type: str = "agglomerative"
        self.additional_model_params = params
        self.metrics = {}
        self.fitted_model = None
        self.hyperparams = None
        self.n_features = None

    def describe(self) -> None:
        """Prints basic information about the model."""
        print(self)

class DBScan_(ClusteringModel):
    def __init__(self, seed = 0, model = None, **params):
        super().__init__(seed, model, **params)
        self.seed: int | float = seed
        self.model = DBSCAN
        self.model_type: str = "dbscan"
        self.additional_model_params = params
        self.metrics = {}
        self.fitted_model = None
        self.hyperparams = None
        self.n_features = None

    def describe(self) -> None:
        """Prints basic information about the model."""
        print(self)


class ClusteringModelOrganizer:
    """
    Class object which can hold instances of several clustering models and can perform hyperparameter optimization, training, and evaluation on sets of them.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        file_directory: str = "",
        models: str | List[str] = {},
        seed: int | float = 0,
        hyperparam_grid: dict = None,
    ):
        """Constructor function for model organizer class object.

        Args:
            df (pd.DataFrame): Dataframe of preprocessed data ready to be fit.
            file_directory (str, optional): File directory where files will be read from and saved to by default. Defaults to "".
            models (str | List[str], optional): Dictionary containing all the models (ClusteringModel objects) held by the organizer. Defaults to {}.
            seed (int | float, optional): Random seed to be used for all probablisitic seeds/random states. Defaults to 0.
            hyperparam_grid (dict, optional): Dictionary of hyperparameter suggestions to be used for hyperparameter tuning. Defaults to None.

        Raises:
            IsADirectoryError: If input file_directory does not exist.
        """
        type_assertion(df, pd.DataFrame), type_assertion(seed, int | float)
        self.seed: int | float = seed
        np.random.seed(self.seed)
        self.x: pd.DataFrame = df
        self.cluster_labels: Dict[str,np.ndarray] = {}
        self.features = self.x.columns
        self.n_samples: int = df.shape[0]
        self.n_features: int = self.x.shape[1]
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
            "k_means": KMeans_,
            "agglomerative": Agglomerative_,
            "mean_shift": MeanShift_,
            "dbscan": DBScan_,
        }

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
            model_type (str): String specifying the type of model to be created. Must be found in the following list: ['k_means', 'agglomerative', 'mean_shift', 'dbscan'].
            name (str, optional): Optional way to return a model based on the name in the self.models dictionary. Must be passed with model_type = None. Defaults to None.

        Raises:
            KeyError: If the model_type does not match one of the possible model types.
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
        self.models[model_type] = self.model_classes[model_type](
                seed = self.seed, random_state = self.seed, **params
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
            assert extension_extract(path) == "pkl"
        except:
            raise FileExistsError(
                "The input file path does not exist in current directory."
            )
        try:
            assert (
                model_type in available_models.keys()
            ) 
        except:
            raise KeyError(f"The model_type must be one of {available_models.keys()}.")

        with open(path, "rb") as file:
            loaded_model = pkl.load(file)
        self.create_model(model_type)
        self.models[model_type].model = loaded_model
        self.models[model_type].fitted_model = loaded_model
        self.models[model_type].n_features = self.n_features

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

    def get_model(self, model_type: str) -> ClusteringModel:
        """Returns direct instance of a Model Class from self.models.

        Args:
            model_type (str, optional): String corresponding to the model type to be returned. Defaults to None.

        Raises:
            ValueError: If an invalid value is passed for either model_type or name or if no value are passed for either.

        Returns:
            ClusteringModel: Class object of the desired model.
        """
        try:
            type_assertion(model_type, str)
            assert model_type in self.models.keys()
        except:
            raise ValueError(
                f"If model_type is passed, it must be a str matching the model_type of an existing model in the self.models dict: {self.models.keys()}. "
            )
        return self.models[model_type]
    
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
        x: pd.DataFrame,
        model: ClusteringModel,
        n_trials: int = 100,
        search_method: str = "base",
    ) -> tuple[int, dict]:
        """Creates and runs an Optuna study using the specified model and the hyperparameter_grid dictionary.

        Args:
            x_train (pd.DataFrame): Feature dataset to be used for training.
            y_train (pd.Series | pd.DataFrame): Target dataset to be used for training.
            model (ClassificationModel): Instance of ClassficationModel class which it to be optimized.
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
                trial, x=x, hyperparam_grid=self.hyperparam_grid
            ),
            n_trials=n_trials,
        )
        return study.best_params


    def predict_set(
        self,
        x: pd.DataFrame,
        models_to_use: list = None,
    ) -> Dict[str, np.ndarray]:
        """Returns dictionary of cluster labels from a set of models.

        Args:
            x (pd.DataFrame): Feature dataset to be used for clustering.
            models_to_use (list, optional): List to specify which models are to be used for clustering. If None is passed, all fitted models will be used. Defaults to None.

        Raises:
            KeyError: If any items in models_to_use do not correspond to an existing model_key.

        Returns:
            Dict[str, np.ndarray]: Dictionary consisting of model_keys and the corresponding model's cluster labels.
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
        cluster_labels: Dict[str, np.ndarray] = {}
        for model_key in models_to_use:
            if self.models[model_key].fitted_model is not None:
                cluster_labels[model_key] = self.models[model_key].predict(x)
        return cluster_labels
   

    def fit_set(
        self,
        models_to_fit: list = None,
        get_scores: bool = False,
        param_tuning: bool = True,
        hyper_params: dict = None,
        n_trials: int = 100,
        optuna_search_method: str = "base",
        **params,
    ) -> None:
        """Fits a set of models

        Args:
            models_to_fit (list, optional): List of model keys to be fit. If None is passed, all existing models will be used. Defaults to None.
            get_scores (bool, optional): If the models are scored after fitting. Defaults to False.
            param_tuning (bool, optional): If the models are to be hyperparameter tuned. Defaults to True.
            n_trials (int, optional): Maximum number of trials to be run in hyperparameter tuning. Defaults to 100.
            hyper_params (dict, optional): Dictionary of hyperparamers to be passed to the models if hyperparameter tuning is not performed. Defaults to None.
            optuna_search_method (str, optional): Method to be used for tuning with optuna, possible options are: ["base", "grid", "random", "gaussian"]. Defaults to "base".
            
        Raises:
            ValueError: If n_trials is an unacceptable value.
            KeyError: If the keys passed in modes_to_fit or in the hyper_params dict do not correspond to an existing model class object.
        """
        type_assertion(get_scores, bool), type_assertion(param_tuning, bool),
        type_assertion(n_trials, int)
        if n_trials <= 0:
            raise ValueError("The value for n_trials must be strictly positive.")
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
            type_assertion(optuna_search_method, str)
            for model_key in models_to_fit:
                model_params[model_key] = self.run_optuna_study(
                    x=self.x,
                    model=self.models[model_key],
                    n_trials=n_trials,
                    search_method=optuna_search_method,
                )
                self.models[model_key].model = available_models[model_key](
                    **model_params[model_key]
                )
        else:
            for model_key in models_to_fit:
                try:
                    assert model_key in hyper_params.keys()
                except:
                    raise KeyError(
                        "Hyperparams must be passed for all models in models_to_fit."
                    )
                model_params[model_key] = hyper_params[model_key]
                self.models[model_key].model = available_models[model_key](
                    **hyper_params[model_key]
                )
                model_class = self.models[model_key]
                model_class.fit(self.x)

                self.models[model_key].hyperparams = model_params[model_key]
        if get_scores is True:
            self.score_set(self.x, models_to_score=models_to_fit)
        for model_key in models_to_fit:
            model_class = self.models[model_key]
            model_class.fit(self.x)
            #model_class.fitted_model = model_class.model
            self.cluster_labels[model_key] = model_class.predict(self.x)
            model_class.hyperparams = model_params[model_key]

    def score_set(
        self,
        x: pd.DataFrame = None,
        cluster_labels: Dict[str, np.ndarray] = None,
        models_to_score: list = None,
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
            x = self.x
        if cluster_labels is None:
            cluster_labels = self.predict_set(x, models_to_score)
        scores: Dict = {}
        for model_key in models_to_score:
            scores[model_key] = self.models[model_key].score(x, cluster_labels[model_key])
            if overwrite_model_metrics is True:
                self.models[model_key].metrics = scores[model_key]
        return scores


    def compare_models(
        self,
        models_to_compare: List[str] = None,
        plot: bool = False,
        display_plot: bool = True,
        save_plot: bool = False,
        save_dir: str = "",
        save_name: str = "clustering_metrics",
    ) -> pd.DataFrame:
        """Compares scoring metrics on a set of fitted models with the option to plot a series of barcharts comparing these metrics. Models must already be fit and scored.

        Args:
            models_to_compare (List[str], optional): List of model_keys corresponding to the models to be compared. If none, all applicable models will be compared. Defaults to None.
            plot (bool, optional): Specifies if the model comparison plot should be created. Defaults to False.
            display_plot (bool, optional): Specifies if the plot will be displayed or not. Defaults to True.
            save_plot (bool, optional): Specifies if the plot should be saved if created. Defaults to False.
            save_dir (str, optional): Directory where the plot is to be saved. Defaults to "".
            save_name (str, optional): Filename the plot will be saved as. Defaults to "classification_metrics".

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
                assert "silhouette_score" in self.models[model_key].metrics.keys()
            except:
                raise KeyError("The models must already be scored to compare them.")
            metrics[model_key] = {
                "Silhouette Score": self.models[model_key].metrics["silhouette_score"],
                "Davies-Bouldin Score ": self.models[model_key].metrics["davies_bouldin_score"],

            }
        metrics_df = pd.DataFrame(metrics).T
        if plot is True:
            plot_num = 1
            fig, axs = plt.subplots(1, 2)
            fig.tight_layout()
            fig.set_size_inches(14, 10)
            for metric in metrics_df.columns:
                plt.subplot(1, 2, plot_num)
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
    
    def visualize_clustering(
            self, 
            models_to_visualize: list = None, 
            dimension_reduction_method: str = 'PCA', 
            plot_grid_width: int = 3, 
            display_plot: bool = True, 
            save_plot: bool = False, 
            save_dir: str = "", 
            save_name: str = "clustering_visualization", 
            **drm_kwargs
            ) -> None:
        """Applies dimensionality reduction on the feature set and plots desired model clusters

        Args:
            models_to_visualize (list, optional): List of models to visualize in a plot grid. If None is passed, all applicable models will be visualized. Defaults to None.
            dimension_reduction_method (str, optional): Method to be used for reducing the dimensionality of the feature space from: ['PCA', 'TSNE']. Defaults to 'PCA'.
            plot_grid_width (int, optional): Number of plots in each row. Defaults to 3.
            display_plot (bool, optional): If True, plot will be displayed. Defaults to True.
            save_plot (bool, optional): If True, plot image will be saved to disk. Defaults to False.
            save_dir (str, optional): Desired directory to save plot image if the plot. Defaults to "".
            save_name (str, optional): Name which the file will be saved with. Defaults to "clustering_visualization".

        Raises:
            KeyError: If models_to_visualize has a key that is not a model in self.models.
        """
        if models_to_visualize is not None:
            type_assertion(models_to_visualize, list)
            try:
                for model_key in models_to_visualize:
                    assert (
                        model_key in self.models.keys()
                        and self.models[model_key].fitted_model is not None
                    )
            except:
                raise KeyError(
                    f"All the values in models_to_visualize must be in {self.models.keys()}"
                )
        else:
            models_to_visualize = [
                model
                for model in self.models.keys()
                if self.models[model].fitted_model is not None
            ]

        if dimension_reduction_method == 'PCA':
            drm = PCA(n_components=2, **drm_kwargs)
        if dimension_reduction_method == 'TSNE':
            drm = TSNE(n_components=2, **drm_kwargs)
        
        x_dr = drm.fit_transform(self.x)
        n_subplots: int = len(models_to_visualize)
        n_cols: int = min(plot_grid_width, n_subplots)
        n_rows: int = math.ceil(n_subplots / plot_grid_width)
        plt.figure(figsize=(n_cols * 5, n_rows * 5))
        for i, model_key in enumerate(models_to_visualize):
            if i > n_subplots:
                pass
            else:
                labs = self.models[model_key].predict(self.x)
                k = len(np.unique(labs))
                plt.subplot(n_rows, n_cols, i + 1)
                plt.scatter(x_dr[:,0], x_dr[:,1], c=labs)
                plt.title(f"{model_key} clustering (k={k})")
        plt.suptitle(f"Clustering Visualization using {dimension_reduction_method}")
        plt.tight_layout()
        if save_plot is True:
            plt.savefig(save_dir + save_name + ".png")
        if display_plot is True:
            plt.show()
