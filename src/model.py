import warnings
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import utils
from .patient_data import BOOL_FEATS, NUM_FEATS, TARGET_NAME, PatientData


# noinspection PyPep8Naming
class EnsembleModel:
    NUM_MODELS = 4

    def __init__(self):
        self.num_folds = None
        self.seed = None

        self.cl_thr = None
        self.scaler = None
        self.models: List[
            Union[
                LogisticRegression,
                DecisionTreeClassifier,
                RandomForestClassifier,
                MLPClassifier,
            ]
        ] = []

        self._trained = False

    def train(
        self,
        data_path: str,
        num_folds: int = 5,
        seed: int = 72,
    ) -> Dict[str, float]:
        logger.info(f"Training Ensemble Model ...")
        self.num_folds = num_folds
        self.seed = seed
        self.cl_thr = None
        self.scaler = None
        self.models = []
        self._trained = False

        # ----------------------------
        # LOADING AND PROCESSING DATA
        # ----------------------------
        df = pd.read_csv(data_path)
        logger.info(f"Dataset shape: {df.shape}")

        X, y = self.preprocess_training_data(df=df)

        logger.info(f"Target distribution:")
        utils.log_multiple_string_obj(df.groupby(TARGET_NAME).size())

        # -----------------------
        # TRAINING K-FOLD MODELS
        # -----------------------
        self.cl_thr = y.sum() / y.shape[0]
        oof_predictions = np.zeros((X.shape[0], self.NUM_MODELS), dtype=np.float32)
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
        for fold_inx, (trn_inx, val_inx) in enumerate(kf.split(y, y)):
            logger.info(f"Training {fold_inx + 1} fold ...")
            trn_X = X[trn_inx]
            val_X = X[val_inx]
            trn_y = y[trn_inx]
            val_y = y[val_inx]

            fold_models = [
                LogisticRegression(random_state=self.seed),
                DecisionTreeClassifier(random_state=self.seed),
                RandomForestClassifier(random_state=self.seed),
                MLPClassifier(random_state=self.seed),
            ]
            for model_inx, model in enumerate(fold_models):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    model = model.fit(X=trn_X, y=trn_y)

                val_y_prob = model.predict_proba(X=val_X)[:, 1]
                val_y_pred = val_y_prob >= self.cl_thr
                logger.info(
                    f"Fold {fold_inx + 1}, {model.__class__.__name__}: "
                    f"MCC={matthews_corrcoef(y_true=val_y, y_pred=val_y_pred):.4f}, "
                    f"F1={f1_score(y_true=val_y, y_pred=val_y_pred):.4f}, "
                    f"ROC_AUC={roc_auc_score(y_true=val_y, y_score=val_y_prob):.4f}"
                )

                oof_predictions[val_inx, model_inx] = val_y_prob
                self.models.append(model)

        y_prob = oof_predictions.mean(axis=-1)
        y_pred = y_prob >= self.cl_thr
        oof_mcc_score = matthews_corrcoef(y_true=y, y_pred=y_pred)
        oof_f1_score = f1_score(y_true=y, y_pred=y_pred)
        oof_roc_auc_score = roc_auc_score(y_true=y, y_score=y_prob)
        logger.info("######")
        logger.info(
            f"Out of Fold, EnsembleModel, "
            f"MCC={oof_mcc_score:.4f}, "
            f"F1={oof_f1_score:.4f}, "
            f"ROC_AUC={oof_roc_auc_score:.4f}"
        )
        logger.info("######")

        self._trained = True
        return {
            "MCC": oof_mcc_score,
            "F1": oof_f1_score,
            "ROC_AUC": oof_roc_auc_score,
        }

    def preprocess_training_data(self, df: pd.DataFrame) -> (np.ndarray, np.ndarray):
        # -------------------
        # PROCESS TARGET COL
        # -------------------
        target_na = df[TARGET_NAME].isna()
        num_target_na = target_na.sum()
        if num_target_na > 0:
            logger.warning(
                f"Target column {TARGET_NAME} has {num_target_na} missing values. "
                f"Excluding rows with missing values from training."
            )
            df = df.loc[~target_na].reset_index(drop=True)
        df[TARGET_NAME] = df[TARGET_NAME].astype(float)

        # ------------------------
        # PROCESS NUMERICAL FEATS
        # ------------------------
        for col_name in NUM_FEATS:
            num_na = df[col_name].isna().sum()
            if num_na > 0:
                logger.warning(
                    f"Column {col_name} has {num_na} missing values. "
                    f"Replacing them with average value."
                )
                df[col_name].fillna(value=df[col_name].mean(), inplace=True)

            df[col_name] = df[col_name].astype(float)

        # ----------------------
        # PROCESS BOOLEAN FEATS
        # ----------------------
        for col_name in BOOL_FEATS:
            num_na = df[col_name].isna().sum()
            if num_na > 0:
                logger.warning(
                    f"Column {col_name} has {num_na} missing values. "
                    f"Replacing them with the most common value."
                )
                df[col_name].fillna(value=df[col_name].value_counts()[0], inplace=True)
            df[col_name] = df[col_name].astype(dtype=float)

        # ----------------------
        # CONSTRUCT X,y MATRICES
        # ----------------------
        X = df[NUM_FEATS + BOOL_FEATS].values
        y = df[TARGET_NAME].values

        # ----------------------
        # SCALING FEATURES
        # ----------------------
        """
        Fitting Standard Scaler. Ideally it should be done within each validation fold, 
        but for simplicity fitting it on the whole dataset here.

        Tree-based methods don't really require the scaling, but it won't harm their
        performance.
        """
        self.scaler = StandardScaler()
        X[:, : len(NUM_FEATS)] = self.scaler.fit_transform(X[:, : len(NUM_FEATS)])

        return X, y

    def data2feat_vector(self, data: PatientData) -> np.ndarray:
        feat_vector = []
        for feat_name in NUM_FEATS:
            feat_vector.append(float(data[feat_name]))

        for feat_name in BOOL_FEATS:
            feat_vector.append(float(data[feat_name]))

        feat_vector = np.asarray(feat_vector)[np.newaxis, :]
        feat_vector[:, : len(NUM_FEATS)] = self.scaler.transform(
            feat_vector[:, : len(NUM_FEATS)]
        )

        return feat_vector

    def predict(self, data: PatientData) -> (float, bool):
        if not self._trained:
            raise ValueError("Model isn't trained yet. Please, train model first.")

        probs = np.empty(shape=len(self.models), dtype=np.float32)
        feat_vector = self.data2feat_vector(data=data)
        for model_inx, model in enumerate(self.models):
            probs[model_inx] = model.predict_proba(X=feat_vector)[0, 1]

        prob = probs.mean()
        pred = prob >= self.cl_thr
        return prob, pred
