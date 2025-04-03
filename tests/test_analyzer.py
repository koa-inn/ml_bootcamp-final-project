path = "/Users/koa/Documents/ML Course/capstone/bootcamp-ml-framework/"
import sys

sys.path.insert(0, path)

import os

# import analyzer
from ml_framework import analyzer
import pytest
import pandas as pd

df = pd.DataFrame({"A": [1, 1, 1, 1, 1], "B": [1, 2, 3, 4, 5], "C": [-2, -1, 0, 1, 2]})

# def test_Analyzer_init():
#     with pytest


def test_set_seed():
    A = analyzer.Analyzer(df)
    with pytest.raises(TypeError):
        A.set_seed("string")


def test_set_target_labels():
    A = analyzer.Analyzer(df)
    A.set_target_labels(["C"])
    assert A.target_labels[0] == "C"
    A.set_target_labels("C")
    assert A.target_labels[0] == "C"
    A.set_target_labels(["B", "C"])
    assert A.target_labels[0] == "B"
    with pytest.raises(TypeError):
        A.set_target_labels(5)
    with pytest.raises(TypeError):
        A.set_target_labels([5])
    with pytest.raises(IndexError):
        A.set_target_labels("D")


def test_shuffle():
    A = analyzer.Analyzer(df)
    df_1 = A.get_frame()
    A.shuffle()
    df_2 = A.get_frame()
    A.shuffle()
    df_3 = A.get_frame()
    assert not df_1.equals(df_2)
    assert not df_1.equals(df_3)
    assert df_1.sum().sum() == df_2.sum().sum()
    with pytest.raises(TypeError):
        A.shuffle(seed="string")


def test_sample():
    A = analyzer.Analyzer(df)
    with pytest.raises(TypeError):
        A.sample(seed="string")
    with pytest.raises(TypeError):
        A.sample(frac="str")
    with pytest.raises(ValueError):
        A.sample(frac=1.2)
    with pytest.raises(ValueError):
        A.sample(frac=-0.2)
    with pytest.raises(ValueError):
        A.sample(n=7)
    with pytest.raises(ValueError):
        A.sample(n=-3)
