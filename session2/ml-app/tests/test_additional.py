import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from data_loader import load_iris_data
from model import IrisClassifier


def test_predict_without_training():
    """Le modèle doit lever une exception si on prédit avant entraînement"""
    clf = IrisClassifier()
    X_train, X_test, y_train, y_test = load_iris_data()
    
    with pytest.raises(Exception):
        clf.predict(X_test[:5])

def test_prediction_shape_consistency():
    """Les prédictions doivent avoir la même taille que l'entrée"""
    clf = IrisClassifier()
    X_train, X_test, y_train, y_test = load_iris_data()
    clf.train(X_train, y_train)

    sample_input = X_test[:10]
    preds = clf.predict(sample_input)
    assert len(preds) == sample_input.shape[0]


def test_model_sanity_check_accuracy():
    """Vérifie que le modèle atteint au moins une précision minimale sur test set"""
    clf = IrisClassifier()
    X_train, X_test, y_train, y_test = load_iris_data()
    clf.train(X_train, y_train)
    accuracy, _ = clf.evaluate(X_test, y_test)
    
    # Sanity check : accuracy > 0.5 (modèle décent)
    assert accuracy >= 0.5
