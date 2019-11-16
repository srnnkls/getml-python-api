# Copyright 2019 The SQLNet Company GmbH

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""
This module contains the predictors for the getml library.
"""

# ------------------------------------------------------------------------------


class _Predictor(object):
    """
    Base class. Should not ever be directly initialized!
    """

    def __init__(self):
        self.thisptr = dict()
        self.thisptr["type_"] = "none"
    
    def _getml_thisptr(self):
        return self.thisptr

    def __repr__(self):
        return self.thisptr.__repr__()

# ------------------------------------------------------------------------------

class LinearRegression(_Predictor):
    """
    Linear regression.

    Simple predictor for regression problems.

    Args:
        learning_rate (float): The learning rate used for numerical training (only
            relevant when categorical features are included).
        reg_lambda (float): L2 regularization parameter.
    """
    def __init__(self, learning_rate=0.9, reg_lambda=1e-10):
        self.thisptr = dict()
        self.thisptr["type_"] = "LinearRegression"
        self.thisptr["lambda_"] = reg_lambda
        self.thisptr["learning_rate_"] = learning_rate

# ------------------------------------------------------------------------------

class LogisticRegression(_Predictor):
    """
    Logistic regression.

    Simple predictor for classification problems.

    Args:
        learning_rate (float) : The learning rate used for the Adaptive Moments algorithm
            (only relevant when categorical features are included).
        reg_lambda (float): L2 regularization parameter.
    """
    def __init__(self, learning_rate=0.9, reg_lambda=1e-10):
        self.thisptr = dict()
        self.thisptr["type_"] = "LogisticRegression"
        self.thisptr["lambda_"] = reg_lambda
        self.thisptr["learning_rate_"] = learning_rate

# ------------------------------------------------------------------------------

class XGBoostClassifier(_Predictor):
    """
    Gradient boosting classifier based on `xgboost <https://xgboost.readthedocs.io/en/latest/>`_.

    Args:
        booster: Which base classifier to use. Possible values: "gbtree", "gblinear" or "dart".
            * "gbtree" are normal gradient boosted decision trees.
            * "gblinear" uses a linear model instead of decision trees.
            * "dart" adds dropout to the standard gradient boosting algorithm.
        colsample_bylevel: Subsample ratio for the columns used, for each level inside a tree.
        colsample_bytree: Subsample ratio for the columns used, for each tree.
        gamma: Minimum loss reduction required for a further split. A higher
            value means stronger regularization.
        learning_rate: Learning rate for the gradient boosting algorithm.
        max_delta_step: The maximum delta step allowed for the weight estimation of each tree.
        max_depth: Maximum allowed depth of the trees.
        min_child_weights: Minimum sum of weights needed in each child node for a split.
        n_estimators: Number of estimators (trees).
        normalize_type: For "dart" booster only. Will be ignored by "gbtree" and "gblinear".
            If set to "tree", then a new tree has the same weight as a single dropped tree.
            If set to "forest", then a new tree has the same weight as a the sum of all dropped trees.
        objective: Objective to be used for the classification problem.
            Possible values: “reg:logistic”, “binary:logistic”, “binary:logitraw”
        n_jobs: Number of parallel threads.
        one_drop: For "dart" booster only. Will be ignored by "gbtree" and "gblinear".
            If set to true, then at least one tree will always be dropped out.
        rate_drop: For "dart" booster only. Will be ignored by "gbtree" and
            "gblinear".  Dropout rate for trees - determines the probability
            that a tree will be dropped out.
        reg_alpha: L1 regularization on the weights.
        reg_lambda: L2 regularization on the weights.
        sample_type: For "dart" booster only. Will be ignored by "gbtree" and "gblinear".
            If set to "uniform", then every tree is equally likely to be dropped out.
            If set to "weighted", then the dropout probability will be proportional to a tree's weight.
        silent: In silent mode, XGBoost will not print out information on the training progress.
        skip_drop: For "dart" booster only. Will be ignored by "gbtree" and "gblinear".
            Probability of skipping the dropout during a given iteration.
        subsample: Subsample ratio from the training set.
    """

    def __init__(
        self,
        booster="gbtree",
        colsample_bylevel=1,
        colsample_bytree=1,
        gamma=0.0,
        learning_rate=0.1,
        max_delta_step=0.0,
        max_depth=3,
        min_child_weight=1.0,
        n_estimators=100,
        normalize_type="tree",
        num_parallel_tree=1,
        n_jobs=1,
        objective="binary:logistic",
        one_drop=False,
        rate_drop=0.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        sample_type="uniform",
        silent=True,
        skip_drop=0.0,
        subsample=1.0
    ):
        super(XGBoostClassifier, self).__init__()

        self.thisptr["type_"] = "XGBoostPredictor"

        self.thisptr["booster_"] = booster
        self.thisptr["colsample_bylevel_"] = colsample_bylevel
        self.thisptr["colsample_bytree_"] = colsample_bytree
        self.thisptr["learning_rate_"] = learning_rate
        self.thisptr["gamma_"] = gamma
        self.thisptr["max_delta_step_"] = max_delta_step
        self.thisptr["max_depth_"] = max_depth
        self.thisptr["min_child_weights_"] = min_child_weight
        self.thisptr["n_estimators_"] = n_estimators
        self.thisptr["normalize_type_"] = normalize_type
        self.thisptr["num_parallel_tree_"] = num_parallel_tree
        self.thisptr["n_jobs_"] = n_jobs
        self.thisptr["objective_"] = objective
        self.thisptr["one_drop_"] = one_drop
        self.thisptr["rate_drop_"] = rate_drop
        self.thisptr["reg_alpha_"] = reg_alpha
        self.thisptr["reg_lambda_"] = reg_lambda
        self.thisptr["sample_type_"] = sample_type
        self.thisptr["silent_"] = silent
        self.thisptr["skip_drop_"] = skip_drop
        self.thisptr["subsample_"] = subsample


# ------------------------------------------------------------------------------

class XGBoostRegressor(_Predictor):
    """
    Gradient boosting regressor based on `xgboost <https://xgboost.readthedocs.io/en/latest/>`_.

    Args:
        booster: Which base classifier to use. Possible values: "gbtree", "gblinear" or "dart".
            * "gbtree" are normal gradient boosted decision trees.
            * "gblinear" uses a linear model instead of decision trees.
            * "dart" adds dropout to the standard gradient boosting algorithm.
        colsample_bylevel: Subsample ratio for the columns used, for each level inside a tree.
        colsample_bytree: Subsample ratio for the columns used, for each tree.
        gamma: Minimum loss reduction required for a further split. A higher value
            means stronger regularization.
        learning_rate: Learning rate for the gradient boosting algorithm.
        max_delta_step: The maximum delta step allowed for the weight estimation of each tree.
        max_depth: Maximum allowed depth of the trees.
        min_child_weights: Minimum sum of weights needed in each child node for a split.
        n_estimators: Number of estimators (trees).
        normalize_type: For "dart" booster only. Will be ignored by "gbtree" and "gblinear".
            If set to "tree", then a new tree has the same weight as a single dropped tree.
            If set to "forest", then a new tree has the same weight as a the sum of all dropped trees.
        n_jobs: Number of parallel threads.
        one_drop: For "dart" booster only. Will be ignored by "gbtree" and "gblinear".
            If set to true, then at least one tree will always be dropped out.
        rate_drop: For "dart" booster only. Will be ignored by "gbtree" and "gblinear".
            Dropout rate for trees - determines the probability that a tree will be dropped out.
        reg_alpha: L1 regularization on the weights.
        reg_lambda: L2 regularization on the weights.
        sample_type: For "dart" booster only. Will be ignored by "gbtree" and "gblinear".
            If set to "uniform", then every tree is equally likely to be dropped out.
            If set to "weighted", then the dropout probability will be proportional to a tree's weight.
        silent: In silent mode, XGBoost will not print out information on the training progress.
        skip_drop: For "dart" booster only. Will be ignored by "gbtree" and "gblinear".
            Probability of skipping the dropout during a given iteration.
        subsample: Subsample ratio from the training set.
    """

    def __init__(
        self,
        booster="gbtree",
        colsample_bylevel=1,
        colsample_bytree=1,
        gamma=0.0,
        learning_rate=0.1,
        max_delta_step=0.0,
        max_depth=3,
        min_child_weight=1.0,
        n_estimators=100,
        normalize_type="tree",
        num_parallel_tree=1,
        n_jobs=1,
        one_drop=False,
        rate_drop=0.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        silent=True,
        sample_type="uniform",
        skip_drop=0.0,
        subsample=1.0
    ):
        super(XGBoostRegressor, self).__init__()

        self.thisptr["type_"] = "XGBoostPredictor"

        self.thisptr["booster_"] = booster
        self.thisptr["colsample_bylevel_"] = colsample_bylevel
        self.thisptr["colsample_bytree_"] = colsample_bytree
        self.thisptr["learning_rate_"] = learning_rate
        self.thisptr["gamma_"] = gamma
        self.thisptr["max_delta_step_"] = max_delta_step
        self.thisptr["max_depth_"] = max_depth
        self.thisptr["min_child_weights_"] = min_child_weight
        self.thisptr["n_estimators_"] = n_estimators
        self.thisptr["normalize_type_"] = normalize_type
        self.thisptr["num_parallel_tree_"] = num_parallel_tree
        self.thisptr["n_jobs_"] = n_jobs
        self.thisptr["objective_"] = "reg:linear"
        self.thisptr["one_drop_"] = one_drop
        self.thisptr["rate_drop_"] = rate_drop
        self.thisptr["reg_alpha_"] = reg_alpha
        self.thisptr["reg_lambda_"] = reg_lambda
        self.thisptr["sample_type_"] = sample_type
        self.thisptr["silent_"] = silent
        self.thisptr["skip_drop_"] = skip_drop
        self.thisptr["subsample_"] = subsample

# ------------------------------------------------------------------------------
