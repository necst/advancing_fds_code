from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import json


def __make_rf(**params):
    """Returns a Random Forest model."""
    return RandomForestClassifier(**params)


def __make_lr(**params):
    """Returns a Logistic Regression model."""
    return LogisticRegression(**params)


def __make_svm(**params):
    """Returns a Support Vector Machine model."""
    return LinearSVC(**params)


def __make_svc(**params):
    """Returns a Support Vector Machine model with the SVC implementation."""
    return SVC(**params)


def __make_calibrated_svm(**params):
    """Returns a Support Vector Machine with class probability estimation"""
    svm = __make_svm(**params)
    return CalibratedClassifierCV(svm)


def __make_xgboost(**params):
    """Returns an XGBoost model."""
    return XGBClassifier(**params)


def __make_neural_network(input_shape, **params):
    from .neural_network import make_network_keras_wrap

    return make_network_keras_wrap(input_shape=input_shape, **params)


def default_builder(
    params, model_type, scale_pos_weight=None, input_shape=(102,), n_jobs=-1, **_
):
    """Reads model configuration from json_path and model_name."""
    
    if "class_weight" in params:
        print("DELETING CLASS WEIGHT!")
        del params["class_weight"]
    
    if model_type == "random_forest":
        return __make_rf(n_jobs=n_jobs, **params)
    elif model_type == "logistic_regression":
        return __make_lr(n_jobs=n_jobs, **params)
    elif model_type == "support_vector_machine":
        return __make_svm(**params)
    elif model_type == "svm_svc":  # scrapped due to long training time
        return __make_svc(n_jobs=n_jobs, **params)
    elif model_type == "calibrated_svm":
        return __make_calibrated_svm(n_jobs=n_jobs, **params)
    elif model_type == "xgboost":
        # return __make_xgboost(scale_pos_weight=scale_pos_weight, n_jobs=n_jobs, **params)
        return __make_xgboost(n_jobs=n_jobs, **params)
    elif model_type == "neural_network":
        return __make_neural_network(input_shape=input_shape, **params)
    else:
        raise ValueError("Model make function not found")


def make_model_from_json(json_path, model_type, build_fn=default_builder, **kwargs):
    """Returns model of type model_type reading the configuration from json_path."""
    with open(json_path, "r") as f:
        params = json.load(f)[model_type]

    print("MAKE MODEL: ", params, model_type, kwargs)
    return build_fn(params, model_type, **kwargs)
