import io
import os
import numpy as np
import pandas as pd
import json

# General utils


def type_assertion(object, type_: type, iterable: bool = False) -> None:
    """Checks if object is of specified type. If it is not, a TypeError is raised.

    Args:
        object (_type_): Object of interest.
        type_ (type): Object type or list of object types to check for.
        iterable (bool, optional): If object is a list or other iterable of objects to be checked, set iterable to True. Defaults to False.

    Raises:
        TypeError: If type of object does not match type_, a type error will be raised.
    """
    if iterable is False:
        try:
            assert isinstance(object, type_)
        except:
            raise TypeError(
                f"Expected object of type {type_}, instead received object of type {type(object)}"
            )
    else:
        for item in object:
            try:
                assert isinstance(item, type_)
            except:
                raise TypeError(
                    f"Expected only objects of type {type_}, instead received object of type {type(item)}"
                )


def extension_extract(filename: str) -> str:
    """Returns the extension of an input filename/path.

    Args:
        filename (str): Filename for which the extension is desired. Str must have form ending in: ".[extension]".

    Returns:
        str: Extension of the filename.
    """
    filename, file_extension = os.path.splitext(filename)
    return file_extension[1:]


def read_to_frame(path: str) -> pd.DataFrame:
    """Reads data file at specified path and returns in a pd.DataFrame.

    Args:
        path (str): File path of data file. Ex: path = '/datasets/data.csv'. List of acceptable extiensions is as follows: ['csv', 'xls', 'xlsx', 'xlsm', 'xlsb', 'odf', 'ods', 'odt', 'pkl', 'xml'].

    Raises:
        FileExistsError: If the file cannot be found at path specified.
        OSError: If the file is of an unacceptable type.

    Returns:
        pd.DataFrame: DataFrame read from specified file.
    """
    type_assertion(path, str)
    try:
        assert os.path.exists(os.path.join(os.getcwd(), path)) == True
    except:
        raise FileExistsError(
            "The input file path does not exist in current directory."
        )
    ext: str = extension_extract(path).lower()
    if ext in ["csv"]:
        df: pd.DataFrame = pd.read_csv(path)
    elif ext in ["xls", "xlsx", "xlsm", "xlsb", "odf", "ods", "odt"]:
        df: pd.DataFrame = pd.read_excel(path)
    elif ext in ["pkl"]:
        df: pd.DataFrame = pd.read_pickle(path)
    elif ext in ["xml"]:
        df: pd.DataFrame = pd.read_xml(path)
    else:
        raise OSError(
            "Filename entered does not have an acceptable filetype. List of acceptable extensions is as follows: ['csv', 'xls', 'xlsx', 'xlsm', 'xlsb', 'odf', 'ods', 'odt', 'pkl', 'xml']."
        )
    return df


def save_to_JSON(dict: dict, path: str = "", name: str = "unnamed") -> None:
    """Saves a dictionary objects a s a JSON file.

    Args:
        dict (dict): Dictionary to be saved.
        path (str): Directory path at which this file will be saved. Defaults to "".
        name (str, optional): Name the file will be saved as. Defaults to "unnamed".
    """
    json_dict = dict
    json_dict = {str(k): str(v) for k, v in json_dict.items()}
    if extension_extract(name) == "json":
        with open(f"{name}", "w") as fp:
            json.dump(dict, fp, indent=4)
    else:
        with open(f"{name}.json", "w") as fp:
            json.dump(dict, fp, indent=4)


def load_JSON(path: str) -> dict:
    """Loads a JSON file and returns a dictionary of the contents.

    Args:
        path (str): Filepath for the JSON file to be loaded.

    Returns:
        dict: Dictionary of JSON file contents.
    """
    if extension_extract(path) == "json":
        with open(f"{path}", "r") as fp:
            data: dict = json.load(fp)
        return data
    else:
        with open(f"{path}.json", "r") as fp:
            data: dict = json.load(fp)
        return data


def create_optuna_suggestion(
    trial, model_type: str, parameter: str, hyperparam_grid: dict
):
    """
    Reads from hyperparameter_grid dictionary and returns the specified parameter as a optuna trial suggestion.
    """
    parameter_dict = hyperparam_grid[model_type][parameter]
    if parameter_dict["type"] == "int":
        kwargs = {
            "name": parameter,
            "low": parameter_dict["low"],
            "high": parameter_dict["high"],
            "log": parameter_dict["log"],
        }
        if parameter_dict["step"] is not None:
            kwargs["step"] = parameter_dict["step"]
        return trial.suggest_int(**kwargs)
    if parameter_dict["type"] == "float":
        kwargs = {
            "name": parameter,
            "low": parameter_dict["low"],
            "high": parameter_dict["high"],
            "log": parameter_dict["log"],
        }
        if parameter_dict["log"] is None:
            kwargs["step"] = parameter_dict["step"]
        return trial.suggest_float(**kwargs)
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


def create_kt_suggestion(hp, model_type, parameter, param_dict):

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
