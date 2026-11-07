"""Built-in scikit-learn model catalog.

The catalog deliberately exposes many small, dependency-free estimators.  It is
not a claim that every estimator wins on every dataset: recipes still choose a
small compatible subset, while power users can select any registered name.
"""

from __future__ import annotations

from kconv.models.sklearn_generic import SklearnSpec

CLASSIFICATION = frozenset({"binary", "multiclass"})
REGRESSION = frozenset({"regression"})


def _spec(
    path: str,
    task_types: frozenset[str],
    *,
    family: str = "linear",
    defaults: dict | None = None,
    seed_param: str | None = "random_state",
) -> SklearnSpec:
    return SklearnSpec(
        path=path,
        task_types=task_types,
        family=family,
        defaults=defaults or {},
        seed_param=seed_param,
    )


SKLEARN_SPECS: dict[str, SklearnSpec] = {
    # Regression: linear and regularized families.
    "linear_regression": _spec("sklearn.linear_model.LinearRegression", REGRESSION),
    "lasso": _spec(
        "sklearn.linear_model.Lasso",
        REGRESSION,
        defaults={"alpha": 0.001, "max_iter": 5000},
    ),
    "elasticnet": _spec(
        "sklearn.linear_model.ElasticNet",
        REGRESSION,
        defaults={"alpha": 0.001, "l1_ratio": 0.5, "max_iter": 5000},
    ),
    "bayesian_ridge": _spec("sklearn.linear_model.BayesianRidge", REGRESSION),
    "huber": _spec(
        "sklearn.linear_model.HuberRegressor",
        REGRESSION,
        defaults={"max_iter": 500},
    ),
    "sgd_regressor": _spec(
        "sklearn.linear_model.SGDRegressor",
        REGRESSION,
        defaults={"max_iter": 2000, "tol": 1e-3},
    ),
    "passive_aggressive_regressor": _spec(
        "sklearn.linear_model.PassiveAggressiveRegressor",
        REGRESSION,
        defaults={"max_iter": 2000, "tol": 1e-3},
    ),
    "knn_regressor": _spec(
        "sklearn.neighbors.KNeighborsRegressor",
        REGRESSION,
        defaults={"n_neighbors": 5, "n_jobs": 1},
        seed_param=None,
    ),
    "decision_tree_regressor": _spec(
        "sklearn.tree.DecisionTreeRegressor",
        REGRESSION,
        family="tree",
        defaults={"max_depth": 8},
    ),
    "extra_tree_regressor": _spec(
        "sklearn.tree.ExtraTreeRegressor",
        REGRESSION,
        family="tree",
        defaults={"max_depth": 8},
    ),
    "random_forest_regressor": _spec(
        "sklearn.ensemble.RandomForestRegressor",
        REGRESSION,
        family="tree",
        defaults={"n_estimators": 120, "max_depth": 10, "n_jobs": 1},
    ),
    "extra_trees_regressor": _spec(
        "sklearn.ensemble.ExtraTreesRegressor",
        REGRESSION,
        family="tree",
        defaults={"n_estimators": 120, "max_depth": 10, "n_jobs": 1},
    ),
    "gradient_boosting_regressor": _spec(
        "sklearn.ensemble.GradientBoostingRegressor",
        REGRESSION,
        family="tree",
        defaults={"n_estimators": 120, "learning_rate": 0.05, "max_depth": 3},
    ),
    "adaboost_regressor": _spec(
        "sklearn.ensemble.AdaBoostRegressor",
        REGRESSION,
        family="tree",
        defaults={"n_estimators": 80},
    ),
    "bagging_regressor": _spec(
        "sklearn.ensemble.BaggingRegressor",
        REGRESSION,
        family="tree",
        defaults={"n_estimators": 40, "n_jobs": 1},
    ),
    "svr": _spec("sklearn.svm.SVR", REGRESSION, seed_param=None),
    "linear_svr": _spec(
        "sklearn.svm.LinearSVR",
        REGRESSION,
        defaults={"max_iter": 5000},
    ),
    "nu_svr": _spec("sklearn.svm.NuSVR", REGRESSION, seed_param=None),
    "kernel_ridge": _spec(
        "sklearn.kernel_ridge.KernelRidge",
        REGRESSION,
        defaults={"alpha": 1.0, "kernel": "rbf"},
        seed_param=None,
    ),
    "pls_regression": _spec(
        "sklearn.cross_decomposition.PLSRegression",
        REGRESSION,
        defaults={"n_components": 2},
        seed_param=None,
    ),
    "ransac": _spec(
        "sklearn.linear_model.RANSACRegressor",
        REGRESSION,
        defaults={"max_trials": 100},
    ),
    "mlp_regressor": _spec(
        "sklearn.neural_network.MLPRegressor",
        REGRESSION,
        defaults={
            "hidden_layer_sizes": (64,),
            "max_iter": 200,
            "early_stopping": True,
        },
    ),
    "dummy_regressor": _spec(
        "sklearn.dummy.DummyRegressor",
        REGRESSION,
        defaults={"strategy": "mean"},
        seed_param=None,
    ),
    "ard_regression": _spec("sklearn.linear_model.ARDRegression", REGRESSION),
    # Classification: neighbors, trees, ensembles and linear/SVM families.
    "knn_classifier": _spec(
        "sklearn.neighbors.KNeighborsClassifier",
        CLASSIFICATION,
        defaults={"n_neighbors": 5, "n_jobs": 1},
        seed_param=None,
    ),
    "decision_tree_classifier": _spec(
        "sklearn.tree.DecisionTreeClassifier",
        CLASSIFICATION,
        family="tree",
        defaults={"max_depth": 8},
    ),
    "extra_tree_classifier": _spec(
        "sklearn.tree.ExtraTreeClassifier",
        CLASSIFICATION,
        family="tree",
        defaults={"max_depth": 8},
    ),
    "random_forest_classifier": _spec(
        "sklearn.ensemble.RandomForestClassifier",
        CLASSIFICATION,
        family="tree",
        defaults={"n_estimators": 120, "max_depth": 10, "n_jobs": 1},
    ),
    "extra_trees_classifier": _spec(
        "sklearn.ensemble.ExtraTreesClassifier",
        CLASSIFICATION,
        family="tree",
        defaults={"n_estimators": 120, "max_depth": 10, "n_jobs": 1},
    ),
    "gradient_boosting_classifier": _spec(
        "sklearn.ensemble.GradientBoostingClassifier",
        CLASSIFICATION,
        family="tree",
        defaults={"n_estimators": 120, "learning_rate": 0.05, "max_depth": 3},
    ),
    "adaboost_classifier": _spec(
        "sklearn.ensemble.AdaBoostClassifier",
        CLASSIFICATION,
        family="tree",
        defaults={"n_estimators": 80},
    ),
    "bagging_classifier": _spec(
        "sklearn.ensemble.BaggingClassifier",
        CLASSIFICATION,
        family="tree",
        defaults={"n_estimators": 40, "n_jobs": 1},
    ),
    "svc": _spec(
        "sklearn.svm.SVC",
        CLASSIFICATION,
        defaults={"gamma": "scale"},
    ),
    "linear_svc": _spec(
        "sklearn.svm.LinearSVC",
        CLASSIFICATION,
        defaults={"max_iter": 5000},
    ),
    "nu_svc": _spec(
        "sklearn.svm.NuSVC",
        CLASSIFICATION,
    ),
    "ridge_classifier": _spec(
        "sklearn.linear_model.RidgeClassifier",
        CLASSIFICATION,
        defaults={"alpha": 1.0},
        seed_param=None,
    ),
    "perceptron": _spec(
        "sklearn.linear_model.Perceptron",
        CLASSIFICATION,
        defaults={"max_iter": 2000},
    ),
    "passive_aggressive_classifier": _spec(
        "sklearn.linear_model.PassiveAggressiveClassifier",
        CLASSIFICATION,
        defaults={"max_iter": 2000},
    ),
    "sgd_classifier": _spec(
        "sklearn.linear_model.SGDClassifier",
        CLASSIFICATION,
        defaults={"max_iter": 2000},
    ),
    "gaussian_nb": _spec(
        "sklearn.naive_bayes.GaussianNB",
        CLASSIFICATION,
        seed_param=None,
    ),
    "bernoulli_nb": _spec(
        "sklearn.naive_bayes.BernoulliNB",
        CLASSIFICATION,
        seed_param=None,
    ),
    "lda": _spec(
        "sklearn.discriminant_analysis.LinearDiscriminantAnalysis",
        CLASSIFICATION,
        seed_param=None,
    ),
    "qda": _spec(
        "sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis",
        CLASSIFICATION,
        defaults={"reg_param": 0.1},
        seed_param=None,
    ),
    "mlp_classifier": _spec(
        "sklearn.neural_network.MLPClassifier",
        CLASSIFICATION,
        defaults={
            "hidden_layer_sizes": (64,),
            "max_iter": 200,
            "early_stopping": True,
        },
    ),
}
