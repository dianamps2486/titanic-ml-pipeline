import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import pytest
from src.train import load_and_preprocess

def test_data_loads():
    X, y = load_and_preprocess("data/titanic.csv")
    assert X.shape[0] > 0, "El dataframe no debe estar vacío"

def test_no_nulls():
    X, y = load_and_preprocess("data/titanic.csv")
    assert X.isnull().sum().sum() == 0, "No deben quedar nulos"

def test_target_binary():
    X, y = load_and_preprocess("data/titanic.csv")
    assert set(y.unique()).issubset({0, 1}), "Target debe ser binario"

def test_feature_count():
    X, y = load_and_preprocess("data/titanic.csv")
    assert X.shape[1] == 7, f"Se esperan 7 features, se tienen {X.shape[1]}"
