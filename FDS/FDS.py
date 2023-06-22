from pathlib import Path
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from preprocessing.aggregation import (
    preprocess_from_past_raw_data,
    preprocess_dataset,
)
from preprocessing.rescaling import standardize_dataset
from preprocessing.const import COLUMN_AGGREGATED_LIST
from typing import List, Optional, Union, NamedTuple, Tuple, Callable
import pandas as pd
from utils.utilities import (
    calc_proportional_class_weight,
    calc_sample_weights,
    remove_features,
    sort_and_reset,
    load_features_from_file,
    load_thresholds_from_json,
)
from enum import Enum
from .model_creation import default_builder, make_model_from_json


class Model(Enum):
    """Learners that can be used with the FDS."""

    RANDOM_FOREST = "random_forest"
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM = "support_vector_machine"  # LinearSVC (Svm with linear kernel and faster computation)
    SVC = "svm_svc"  # svm (SVC implementation)
    CALIBRATED_SVM = "calibrated_svm"  # LinearSVC with CalibratedClassifierCV for class probability estimation
    XGBOOST = "xgboost"
    NEURAL_NETWORK = "neural_network"


class UpdatePolicy(Enum):
    """FDS update policies."""

    ONE_WEEK = "1week"
    TWO_WEEKS = "2weeks"
    ONE_MONTH = "1month"


class FDSDataset(NamedTuple):
    """Data format for the FDS."""

    raw: pd.DataFrame
    aggr: pd.DataFrame
    std: pd.DataFrame


# first timestamp in the 2014-2015 dataset
STARTING_DATASET_TS = pd.Timestamp(year=2014, month=10, day=22)
MAX_TS_2014_15 = pd.Timestamp(year=2015, month=3, day=1, hour=23, minute=59)


# list of all the timestamps for each update policy
UPDATE_TS = {
    UpdatePolicy.ONE_WEEK: [
        pd.Timestamp(year=2015, month=1, day=5),
        pd.Timestamp(year=2015, month=1, day=12),
        pd.Timestamp(year=2015, month=1, day=19),
        pd.Timestamp(year=2015, month=1, day=26),
        pd.Timestamp(year=2015, month=2, day=2),
        pd.Timestamp(year=2015, month=2, day=9),
        pd.Timestamp(year=2015, month=2, day=16),
        pd.Timestamp(year=2015, month=2, day=23),
    ]
    + [MAX_TS_2014_15],
    UpdatePolicy.TWO_WEEKS: [
        pd.Timestamp(year=2015, month=1, day=5),
        pd.Timestamp(year=2015, month=1, day=19),
        pd.Timestamp(year=2015, month=2, day=2),
        pd.Timestamp(year=2015, month=2, day=16),
    ]
    + [MAX_TS_2014_15],
    UpdatePolicy.ONE_MONTH: [
        pd.Timestamp(year=2015, month=1, day=1),
        pd.Timestamp(year=2015, month=2, day=1),
    ]
    + [MAX_TS_2014_15],
}


class FraudDetectionSystem:
    """
    Class that simulates a Fraud Detection System with a scikit-learn like API.

    :param features: FDS features
    :type features: List[str]
    :param update_policy: FDS update policy (1week, 2weeks, ...)
    :type update_policy: Union[UpdatePolicy, str]
    :param model_type: Name of model used by the FDS
    :type model_type: Model
    :param save_data_path: path where to save data used in simulation (rejected, accepted transactions...)
    :type save_data_path: Union[pathlib.Path, str]
    :param update_ts_list: List of training timestamps
    :type update_ts_list: List[pandas.Timestamp]
    :param last_fit_ts: Last fitting timestamp
    :type last_fit_ts: pandas.Timestamp
    :param curr_iteration: Current training iteration, position in update_ts_list
    :type curr_iteration: int
    :param threshold: Threshold for classification for given model - defaults to 0.5
    :type threshold: float
    :param make_model_func: ML model builder function, returns model given name
    :type make_model_func: Callable
    :param hyperparam_file: Path of selected hyperparameters for each model
    :type hyperparam_file: pathlib.Path
    :param model: Model returned by make_model_func
    :type model: Any
    :param scaler_cache_dir: Name of the cache directory where standard scalers are saved
    :type scaler_cache_dir: str
    """

    def __init__(
        self,
        model_type: Union[Model, str],
        update_policy: Union[UpdatePolicy, str],
        save_data_path: Union[Path, str],
        hyperparam_file: Path = "",
        feature_dir: Path = "",
        thresholds_file: Optional[Path] = "",
        make_model_func: Callable = default_builder,
        scaler_cache_dir: str = "fds",
        n_jobs: int = -1
    ):
        model_type = Model(model_type)
        update_policy = UpdatePolicy(update_policy)
        self.feature_dir = feature_dir
        self.features = self.select_feature_set(model_type)
        self.update_policy = update_policy
        self.model_type = model_type
        self.save_data_path = save_data_path
        self.update_ts_list = self.select_update_list(update_policy)
        self.last_fit_ts = None
        self.curr_iteration = None
        if thresholds_file is not None:
            all_thresholds = load_thresholds_from_json(thresholds_file)
        else:
            all_thresholds = {}
        self.threshold = all_thresholds.get(
            model_type.value, None
        )
        self.make_model_func = make_model_func
        self.hyperparam_file = hyperparam_file
        self.model = None
        self.scaler_cache_dir = scaler_cache_dir
        self.use_threshold = self.threshold is not None
        self.n_jobs = n_jobs

    def __repr__(self):
        repr = "*" * 39 + "FDS" + "*" * 38 + "\n"
        repr += (
            "PARAMS_FILE: %s\n"
            "UPDATE_POLICY: %s\n"
            "MODEL: %s\n"
            "THRESHOLD: %s\n"
            "FEATURES: %s"
            % (
                self.hyperparam_file,
                self.update_policy.value,
                self.model_type.value,
                self.threshold,
                ", ".join(self.features[:3] + ["..."] + self.features[-3:]),
            )
        )
        repr += "\n" + "*" * 80
        return repr

    @staticmethod
    def select_update_list(
        update_policy: Union[UpdatePolicy, str]
    ) -> List[pd.Timestamp]:
        """
        Returns the list of training timestamps for the given update policy.

        :param update_policy: Fraud Detection System update policy (1week, 2weeks, ...)
        :type update_policy: Union[UpdatePolicy, str]
        :return: List of timestamps
        :rtype: List[pandas.Timestamp]
        """
        update_policy = UpdatePolicy(update_policy)
        return UPDATE_TS[update_policy]

    def select_feature_set(
        self, model_type: Union[Model, str], feature_file: Optional[Path] = None
    ) -> List[str]:
        """
        Returns the list of training timestamps for the given update policy.

        :param model_type: Name of the model or enum constant
        :type model_type: Union[Model, str]
        :return: List of features
        :rtype: List[str]
        """
        model_type = Model(model_type)
        default_file: Path = self.feature_dir / (model_type.value + ".txt")
        if feature_file is None:
            feature_file = default_file 

        return load_features_from_file(feature_file)

    def update_iteration(self):
        """
        Updates FDS fitting iteration.
        I've added this function to track the current training iteration over
        the dataset. Datasets span over a period of around 2 months and they
        are divided by fixed lists of timestamps, one for each update policy.
        """
        self.curr_iteration = (
            0 if self.curr_iteration is None else self.curr_iteration + 1
        )

    def is_fitting_time(self, train_ts: pd.Timestamp) -> bool:
        """
        This function tells if the FDS can be trained.

        :param train_ts: current simulation timestamp
        :type train_ts: pandas.Timestamp
        :return: result of check
        :rtype: bool
        """

        if not self.get_last_fit_ts():
            print("Model not initialized.")
            return True

        if train_ts < self.get_last_fit_ts():
            print("Model already fit. Skipping")
            return False
        elif train_ts < self.get_next_fit_ts():
            return False
        else:
            return True

    def get_last_fit_ts(self) -> Optional[pd.Timestamp]:
        """Get timestamp of last fitting.

        :return: Last fitting timestamp
        :rtype: Optional[pandas.Timestamp]
        """
        if self.curr_iteration == None:
            return None

        return self.update_ts_list[self.curr_iteration]

    def get_next_fit_ts(self) -> pd.Timestamp:
        """Get timestamp of the next fitting.

        :return: Next fitting timestamp
        :rtype: Optional[pandas.Timestamp]
        """
        next_it = self.curr_iteration + 1

        if next_it == len(self.update_ts_list):
            next_it = self.curr_iteration

        return self.update_ts_list[next_it]

    def fit(
        self, data: FDSDataset, train_ts: pd.Timestamp, sample_weights=None, **kwargs
    ):
        """
        Fits base model with training set and sample weights.
        Precalculated sample weights can be passed by the parameter, otherwise
        the model makes them by default.
        **If data.std is None then the fitting function calculates it from the
        aggregated dataset and using the scaler saved in self.scaler_cache_dir.**
        At last, it updates the model training iteration.

        :param data: Training datasets
        :type data: FDS.Data
        :param train_ts: Current training timestamp
        :type train_ts: pandas.Timestamp
        :param sample_weights: List of weights associated to samples
        :type sample_weights: Optional[List[float]]
        :kwargs: extra arguments passed to self.make_model_func
        """
        print("Start fitting...")

        if data.std is None:
            fds_aggr, fds_std = self.compute_aggr_std_df(data.aggr, fit=True)
            prepared_data = FDSDataset(data.raw, fds_aggr, fds_std)
        else:
            prepared_data = data

        features_w_class = self.features + ["Fraud"]
        features_w_class_ts = self.features + ["Fraud", "Timestamp"]

        # if the data isn't with tthe right features
        training_data = remove_features(prepared_data.std, features_w_class_ts)

        x_neg = training_data[training_data["Fraud"] == 0]
        x_pos = training_data[training_data["Fraud"] == 1]
        pos_weight = x_neg.shape[0] / x_pos.shape[0]
        # -2 due to additional 'Fraud' and 'Timestamp' columns
        input_shape = (training_data.shape[1] - 2,)

        self.model = make_model_from_json(
            self.hyperparam_file,
            self.model_type.value,
            build_fn=self.make_model_func,
            input_shape=input_shape,
            scale_pos_weight=pos_weight,
            n_jobs=self.n_jobs,
            **kwargs,
        )

        assert self.model is not None

        if sample_weights is None:
            print("computing sample weights.")
            sample_weights = self.compute_sample_weights(data, train_ts)

        df_train = remove_features(training_data, features_w_class)

        y_train = df_train["Fraud"]
        x_train = df_train.drop(["Fraud"], axis=1)
        x_train = x_train[self.features]

        # if self.model_type == Model.SVM:
        #     self.model = CalibratedClassifierCV(self.model, n_jobs=-1)

        if (
            self.model_type == Model.NEURAL_NETWORK
        ):
            class_weight = calc_proportional_class_weight(training_data["Fraud"])
            weights = np.array(sample_weights)
            fit_kwargs = {
                "sample_weight": weights,
                "class_weight": class_weight,
            }
        else:
            fit_kwargs = {"sample_weight": sample_weights}

        self.model.fit(x_train, y_train, **fit_kwargs)

        self.update_iteration()
        print("Fitting complete.")

    def compute_sample_weights(self, training_data, train_ts) -> List:
        """
        Returns calculated sample weights.

        :param training_data: FDS training data
        :type training_data: FDS.Data
        :param train_ts: Current training timestamp
        :type train_ts: pandas.Timestamp
        :return: List of sample weights
        :rtype: List[float]
        """
        return calc_sample_weights(training_data.aggr, train_ts)

    def predict(
        self, transactions: pd.DataFrame
    ) -> Union[Tuple[int, float], Tuple[np.ndarray, np.ndarray]]:
        """
        Returns classification and probability for one or more aggregated
        transactions.

        :param transactions: one or more aggregated transactions.
        :type transactions: pandas.DataFrame
        :return: prediction, probability
        :rtype: Tuple[int, float] or Tuple[list, list]
        """
        assert self.model is not None
        transactions = transactions[self.features]
        
        try:
            predictions_prob = self.model.predict_proba(transactions)[:, 1]
            predictions_prob = np.array(predictions_prob)
        except AttributeError as e:
            predictions_prob = self.model.predict(transactions)
            if self.use_threshold:
                print("USING THRESHOLD ON MODEL THAT DOES NOT HAVE predict_proba")
            
        if self.use_threshold:
            predictions = predictions_prob > self.threshold
        else:
            predictions = self.model.predict(transactions)

        if transactions.shape[0] == 1:
            return predictions[0], predictions_prob[0]
        else:
            return predictions.reshape(-1), predictions_prob.reshape(-1)

    def predict_raw(
        self,
        transactions,
        past_user_raw_data,
        disable_tqdm=True,
        n_jobs=0,
        return_aggregated=False,
    ):
        """
        Returns classification for one or more raw transactions and past user data.

        :param transactions: transactions to classify
        :type transactions: pandas.DataFrame
        :param past_user_raw_data: past user raw transactions
        :type past_user_raw_data: pandas.DataFrame
        :param disable_tqdm: disable the nice progress bar
        :type disable_tqdm: bool
        :param n_jobs: how many cpus to use, -1 for all available
        :type n_jobs: int
        :param return_aggregated: return aggregated input transactions
        :type return_aggregated: bool
        :return: labels and score
        :rtype: Tuple[int, float] or Tuple[list, list]
        """
        transactions_cp = transactions.copy(deep=True)

        # past_user_raw_data_cp.loc[:, "Fraud"] = 0
        transactions_cp.loc[:, "Fraud"] = 0

        assert "Fraud" in transactions_cp.columns
        assert "Fraud" in past_user_raw_data.columns

        preprocessed_data_all_features = preprocess_from_past_raw_data(
            transactions_cp,
            past_user_raw_data,
            disable_tqdm=disable_tqdm,
            n_jobs=n_jobs,
        )

        model_features = self.features
        preprocessed_data = remove_features(
            preprocessed_data_all_features, model_features
        )
        preprocessed_data = preprocessed_data[model_features]

        standardized_trans = standardize_dataset(
            preprocessed_data,
            fit=False,
            save_folder=self.scaler_cache_dir,
            disable_tqdm=disable_tqdm,
            n_jobs=n_jobs,
        )

        if return_aggregated:
            return (
                self.predict(standardized_trans),
                preprocessed_data_all_features,
            )
        else:
            return self.predict(standardized_trans)

    def reprocess_with_new_data(
        self, old_data: FDSDataset, new_data_raw: pd.DataFrame, **kwargs
    ) -> FDSDataset:
        """
        Extracts old user data from dataset and reprocesses new data with it.

        :param old_data: Some data
        :type old_data: FDS.Data
        :param new_data_raw: New raw data to be preprocessed with old data
        :type new_data_raw: pandas.DataFrame
        :param kwargs: extra arguments for preprocess_dataset function
        :return: new preprocessed data
        :rtype: FDS.Data
        """
        print("Reprocessing dataset with new data.")

        if new_data_raw.empty:
            print("No new data. Returning data as is.")
            return old_data

        initial_shape = new_data_raw.shape[0]

        new_data_raw = new_data_raw.loc[
            ~new_data_raw.TransactionID.isin(old_data.raw.TransactionID)
        ]

        duplicates = initial_shape - new_data_raw.shape[0]

        if duplicates > 0:
            print(
                "[WARNING]: dropped %d duplicate transactions from new data"
                % duplicates
            )

        # get all users that have produced new data
        to_recalc = old_data.raw.UserID.isin(new_data_raw.UserID)

        # divide dataset in two parts: find the part that needs to be recalculated
        # and the other that does not

        # part that must be recalculated
        dataset_raw_to_recalc = old_data.raw.loc[to_recalc]

        # find from aggr and std dataset all transaction IDs the part that does not
        # need recalculation
        dataset_aggr_no_recalc = old_data.aggr.loc[
            ~old_data.aggr.TransactionID.isin(dataset_raw_to_recalc.TransactionID)
        ]

        # consider only raw columns
        dataset_raw_to_recalc = dataset_raw_to_recalc[new_data_raw.columns]
        dataset_raw_to_recalc = pd.concat([dataset_raw_to_recalc, new_data_raw])
        new_preprocessed = preprocess_dataset(dataset_raw_to_recalc, **kwargs)

        new_preprocessed = new_preprocessed[old_data.aggr.columns]

        # merge, sort and reset indices
        new_aggr_recalculated = pd.concat([dataset_aggr_no_recalc, new_preprocessed])
        new_aggr_recalculated = sort_and_reset(new_aggr_recalculated)
        new_aggr_recalculated = new_aggr_recalculated[
            set(COLUMN_AGGREGATED_LIST + self.features)  # type: ignore
        ]

        new_raw = pd.concat([old_data.raw, new_data_raw])
        new_raw = sort_and_reset(new_raw)

        print("Finished reprocessing dataset.")
        return FDSDataset(new_raw, new_aggr_recalculated, None)  # type: ignore

    def reprocess_users(
        self, data: FDSDataset, users: List[str], fit=False, **kwargs
    ) -> FDSDataset:
        """
        Given a dataset it recalculates aggregated features for a list of users.

        :param Data: data to be preprocessed
        :type Data: FDS.Data
        :param users: list of UserID of users to be reprocessed
        :type users: List[str]
        :param fit: true to fit standard scaler
        :type fit: bool
        :param kwargs: extra arguments for preprocess_dataset function
        :return: reprocessed data
        :rtype: FDS.Data
        """
        aggr_to_recalc = data.aggr.loc[data.aggr.UserID.isin(users)]
        aggr_no_reprocess = data.aggr.loc[~data.aggr.UserID.isin(users)]

        aggr_reprocessed = preprocess_dataset(aggr_to_recalc, **kwargs)
        aggr_reprocessed = aggr_reprocessed[aggr_no_reprocess.columns]

        new_aggr = pd.concat([aggr_no_reprocess, aggr_reprocessed])
        new_aggr = sort_and_reset(new_aggr)
        new_aggr = new_aggr[set(COLUMN_AGGREGATED_LIST + self.features)]  # type: ignore
        return FDSDataset(data.raw, new_aggr, None)  # type: ignore

    def compute_aggr_std_df(
        self, df_aggregated: pd.DataFrame, fit: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Removes feature not in self.features, aggregates and standardizes dataset.
        :param df_aggregated: aggregated transactions dataset
        :type df_aggregated: pandas.Dataframe
        :param fit: true to fit standard scaler
        :type fit: bool
        :return: aggregated and standardized datasets
        :rtype: Tuple[pandas.DataFrame, pandas.DataFrame]
        """

        df_aggregated = remove_features(
            df_aggregated, set(COLUMN_AGGREGATED_LIST + self.features)
        )

        df_standardized = standardize_dataset(
            df_aggregated, fit=fit, save_folder=self.scaler_cache_dir
        )

        return df_aggregated, df_standardized
