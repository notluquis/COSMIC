"""
Module for clustering astronomical data using HDBSCAN with optional
grid search or Optuna-based hyperparameter tuning.
"""
import datetime
import hdbscan
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from astropy.table import QTable, vstack
from scipy.stats import iqr
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV
from hdbscan.validity import validity_index
from typing import Sequence, Any

# Configure global plotting style
mpl.rcParams.update({
    'figure.figsize':      (8, 6),
    'figure.dpi':          100,
    'font.size':           12,
    'axes.titlesize':      14,
    'axes.labelsize':      12,
    'axes.edgecolor':      'black',
    'axes.linewidth':      1.0,
    'axes.grid':           True,
    'grid.linestyle':      '--',
    'grid.linewidth':      0.5,
    'grid.color':          '0.8',
    'xtick.direction':     'in',
    'ytick.direction':     'in',
    'xtick.top':           True,
    'ytick.right':         True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'legend.frameon':      False,
    'legend.fontsize':     10,
})

class FullSplit:
    """Use entire dataset for both training and testing in GridSearchCV."""
    def split(self, X, y=None, groups=None):
        n = X.shape[0]
        yield np.arange(n), np.arange(n)

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

def _compute_relative_validity_from_mst(mst, labels):
    """
    Recompute HDBSCAN relative_validity_ given a minimum spanning tree (MST)
    and possibly-pruned labels.
    """
    sizes = np.bincount(labels + 1)
    noise_size = sizes[0]
    cluster_sizes = sizes[1:]
    total_points = noise_size + cluster_sizes.sum()
    n_clusters = len(cluster_sizes)

    # Initialize arrays for intra- and inter-cluster distances
    dsc = np.zeros(n_clusters)
    dspc = np.full(n_clusters, np.inf)
    max_dist = 0.0
    min_outlier_dist = np.inf

    # Iterate MST edges
    if hasattr(mst, 'iterrows'):
        # pandas DataFrame
        for _, row in mst.iterrows():
            frm, to, dist = int(row['from']), int(row['to']), row['distance']
            lab_a, lab_b = labels[frm], labels[to]
            max_dist = max(max_dist, dist)
            if lab_a == -1 and lab_b == -1:
                continue
            if lab_a == -1 or lab_b == -1:
                min_outlier_dist = min(min_outlier_dist, dist)
                continue
            if lab_a == lab_b:
                dsc[lab_a] = max(dsc[lab_a], dist)
            else:
                dspc[lab_a] = min(dspc[lab_a], dist)
                dspc[lab_b] = min(dspc[lab_b], dist)
    else:
        # numpy ndarray
        for frm, to, dist in mst:
            frm, to = int(frm), int(to)
            lab_a, lab_b = labels[frm], labels[to]
            max_dist = max(max_dist, dist)
            if lab_a == -1 and lab_b == -1:
                continue
            if lab_a == -1 or lab_b == -1:
                min_outlier_dist = min(min_outlier_dist, dist)
                continue
            if lab_a == lab_b:
                dsc[lab_a] = max(dsc[lab_a], dist)
            else:
                dspc[lab_a] = min(dspc[lab_a], dist)
                dspc[lab_b] = min(dspc[lab_b], dist)

    # Final adjustments for relative validity
    if min_outlier_dist == np.inf:
        min_outlier_dist = max_dist
    correction = 2 * (max_dist if n_clusters > 1 else min_outlier_dist)
    dspc[np.isinf(dspc)] = correction
    V = (dspc - dsc) / np.maximum(dspc, dsc)
    return np.sum(cluster_sizes * V / total_points)

class HDBSCANEstimator(BaseEstimator, ClusterMixin):
    """
    sklearn-style wrapper around hdbscan.HDBSCAN that supports:
      - external persistence_threshold for pruning low-persistence clusters
      - recomputation of relative_validity_ after pruning
    """
    def __init__(self,
                 *,
                 min_cluster_size: int = 10,
                 min_samples: int | None = None,
                 persistence_threshold: float = 0.0,
                 **hdbscan_kwargs):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.hdbscan_kwargs = hdbscan_kwargs
        self.persistence_threshold = persistence_threshold
        self.model_ = None

    def fit(self, X, y=None):
        """Fit HDBSCAN to X."""
        params = {
            'min_cluster_size': self.min_cluster_size,
            'min_samples': self.min_samples,
            'gen_min_span_tree': True,
            'core_dist_n_jobs': -1,
            **self.hdbscan_kwargs
        }
        self.model_ = hdbscan.HDBSCAN(**params).fit(X)
        return self

    def predict(self, X):
        """Assign labels to new points (same as fit, since no new training)."""
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict().")
        return self.model_.labels_

    def score(self, X, y=None):
        """
        Return relative_validity_, recomputed if pruning by persistence_threshold.
        """
        if self.model_ is None:
            raise RuntimeError("Must call fit() before score().")

        t = self.persistence_threshold
        labels = self.model_.labels_.copy()
        pers   = self.model_.cluster_persistence_

        # Extract MST as numpy array
        mst = self.model_.minimum_spanning_tree_.to_pandas()[['from','to','distance']].to_numpy()

        # Prune clusters below threshold
        if t > 0.0:
            for cid, pval in enumerate(pers):
                if pval < t:
                    labels[labels == cid] = -1

        # Return built-in or recalculated validity
        if t <= 0.0:
            return float(self.model_.relative_validity_)
        else:
            return float(_compute_relative_validity_from_mst(mst, labels))

class Clustering:
    """
    High-level class to perform clustering on an astropy QTable,
    optionally tuning hyperparameters via grid search or Optuna.
    """
    def __init__(
        self,
        data: QTable,
        bad_data: QTable | None = None,
        *,
        search_method: str = 'grid',
        sqlite_path: str = "optuna_study.db",
        study_name: str = None
    ):
        if search_method not in ('grid', 'optuna'):
            raise ValueError("search_method must be 'grid' or 'optuna'.")
        self.search_method = search_method
        self.data = data
        self.bad_data = bad_data
        self.storage_url = f"sqlite:///{sqlite_path}" if sqlite_path else None
        self.study_name = study_name or f"study_{datetime.datetime.now():%Y%m%d_%H%M%S_%f}"
        # to be filled after search
        self.clusterer = None
        self.cv_results_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.combined_data = None
        self._study = None
        self._pareto_trials = []

    def search(
        self,
        columns: list[str],
        *,
        persistence_threshold: float = 0.0,
        param_grid: dict[str, list] | None = None,
        grid_kwargs: dict | None = None,
        optuna_search_space: dict[str, dict] | None = None,
        n_trials: int = 50,
        sampler: str = 'TPESampler',
        sampler_kwargs: dict | None = None,
        score_method: str | list[str] = 'relative_validity',
        hdbscan_kwargs: dict | None = None,
    ):
        """Run grid or Optuna search over given columns, then annotate clusters."""
        X = self.data[columns].to_pandas().values
        hdbscan_kwargs = hdbscan_kwargs or {}

        if self.search_method == 'grid':
            self._run_grid_search(X, param_grid, grid_kwargs,
                                  persistence_threshold=persistence_threshold,
                                  hdbscan_kwargs=hdbscan_kwargs)
        else:
            self._run_optuna_search(X,
                                    optuna_search_space,
                                    persistence_threshold=persistence_threshold,
                                    n_trials=n_trials,
                                    sampler_name=sampler,
                                    sampler_kwargs=sampler_kwargs,
                                    score_method=score_method,
                                    hdbscan_kwargs=hdbscan_kwargs)

        # Annotate results back into the QTable
        self.data['cluster'] = self.clusterer.labels_
        self.data['probability_hdbscan'] = self.clusterer.probabilities_
        if self.bad_data is not None:
            combo = vstack([self.data, self.bad_data])
            combo['cluster'].fill_value = -1
            combo['probability_hdbscan'].fill_value = 0.0
            self.combined_data = combo
        else:
            self.combined_data = self.data.copy()

    def _run_grid_search(self, X, param_grid, grid_kwargs,
                         *, persistence_threshold, hdbscan_kwargs):
        """Internal: grid search over HDBSCANEstimator."""
        est = HDBSCANEstimator(
            persistence_threshold=persistence_threshold,
            **hdbscan_kwargs
        )
        gs = GridSearchCV(
            est,
            param_grid or {'min_cluster_size': [5, 10, 20]},
            scoring=lambda e, X: e.score(X),
            cv=FullSplit(),
            n_jobs=-1,
            verbose=1,
            error_score=float('-inf'),
            **(grid_kwargs or {})
        )
        gs.fit(X)
        self.clusterer = gs.best_estimator_.model_
        self.cv_results_ = gs.cv_results_
        self.best_params_ = gs.best_params_
        self.best_score_ = gs.best_score_

    def _run_optuna_search(
        self,
        X: np.ndarray,
        search_space: dict[str, dict] | None,
        *,
        persistence_threshold: float,
        n_trials: int,
        sampler_name: str,
        sampler_kwargs: dict | None,
        score_method: str | list[str],
        hdbscan_kwargs: dict
    ):
        search_space    = search_space or {}
        sampler_kwargs  = sampler_kwargs or {}
        methods         = [score_method] if isinstance(score_method, str) else score_method

        # 1) Dynamically select the sampler class from optuna.samplers
        import optuna.samplers as _samplers_module
        
        # Obtener todos los nombres de clases de sampler disponibles en optuna.samplers
        _available_samplers = [
            name for name, obj in vars(_samplers_module).items()
            if isinstance(obj, type) and name.endswith("Sampler")
        ]
        
        try:
            SamplerClass = getattr(_samplers_module, sampler_name)
        except AttributeError:
            raise ValueError(
                f"Unknown sampler '{sampler_name}'.\n"
                f"Available samplers are: {', '.join(_available_samplers)}"
            )

        # 1a) Special handling for GridSampler: infer grid_space from low/high or use explicit choices
        if sampler_name == 'GridSampler':
            grid_space: dict[str, list] = {}
            for param, info in search_space.items():
                if 'choices' in info:
                    grid_space[param] = info['choices']
                else:
                    if info.get('type') == 'int':
                        low, high = info['low'], info['high']
                        grid_space[param] = list(range(low, high + 1))
                    elif info.get('type') == 'float':
                        low, high = info['low'], info['high']
                        grid_space[param] = list(np.linspace(low, high, num=10))
                    else:
                        raise ValueError(
                            f"Cannot infer grid for parameter '{param}' of tipo '{info.get('type')}'"
                        )
            sampler_kwargs.setdefault('search_space', grid_space)

        # 1b) Special handling for GPSampler: default independent_sampler to RandomSampler if not provided
        if sampler_name == 'GPSampler':
            from optuna.samplers import RandomSampler
            seed = sampler_kwargs.get('seed', None)
            sampler_kwargs.setdefault(
                'independent_sampler',
                RandomSampler(seed=seed)
            )

        # Instantiate the sampler with final kwargs
        sampler = SamplerClass(**sampler_kwargs)
        # 3) Creamos el estudio, pasándole storage + study_name + load_if_exists
        if self.storage_url is None:
            # Si no se dio sqlite_path, usamos la base en memoria (por defecto)
            if len(methods) > 1:
                study = optuna.create_study(
                    study_name=self.study_name,
                    load_if_exists=False,
                    directions=['maximize'] * len(methods),
                    sampler=sampler
                )
            else:
                study = optuna.create_study(
                    study_name=self.study_name,
                    load_if_exists=True,
                    direction='maximize',
                    sampler=sampler
                )
        else:
            # Usamos el mismo archivo sqlite para guardar/recuperar
            if len(methods) > 1:
                study = optuna.create_study(
                    study_name=self.study_name,
                    storage=self.storage_url,
                    load_if_exists=True,
                    directions=['maximize'] * len(methods),
                    sampler=sampler
                )
            else:
                study = optuna.create_study(
                    study_name=self.study_name,
                    storage=self.storage_url,
                    load_if_exists=True,
                    direction='maximize',
                    sampler=sampler
                )

        self._study = study

        # 3) Define the objective function
        def objective(trial):
            # 1) Sugerimos los hiperparámetros definidos en search_space
            params: dict[str, Any] = {}
            for name, space in (search_space or {}).items():
                try:
                    if 'choices' in space:
                        params[name] = trial.suggest_categorical(name, space['choices'])
                    elif space['type'] == 'int':
                        params[name] = trial.suggest_int(
                            name, space['low'], space['high'], log=space.get('log', False)
                        )
                    elif space['type'] == 'float':
                        params[name] = trial.suggest_float(
                            name, space['low'], space['high'], log=space.get('log', False)
                        )
                    else:
                        raise ValueError(f"Unknown parameter for '{name}': {space}")
                except Exception as e:
                    print(f"  [ERROR] al sugerir parámetro '{name}': {e!r}")
                    raise
        
            # 2) Creamos y ajustamos el HDBSCANEstimator
            try:
                estimator = HDBSCANEstimator(
                    min_cluster_size=params.get('min_cluster_size', None),
                    min_samples=params.get('min_samples', None),
                    persistence_threshold=persistence_threshold,
                    **hdbscan_kwargs
                )
                estimator.fit(X)
            except Exception as e:
                print(f"  [ERROR] al entrenar HDBSCAN con params {params}: {e!r}")
                # Hacemos fallar explícitamente este trial
                raise
        
            # Tras el fit, imprimimos cuántos clusters detectó (antes de poda)
            results: list[float] = []
        
            for m in methods:
                if m == 'relative_validity':
                    # llamamos a score(), que internamente aplica persistence_threshold
                    try:
                        rv = estimator.score(X)
                        if np.isnan(rv):
                            print(f"  [RV] Trial #{trial.number}: score() devolvió NaN. Lo convierto a -inf.")
                            rv = float('-inf')
                        else:
                            rv = float(rv)
                    except Exception as e:
                        print(f"  [RV] Trial #{trial.number}: excepción en score(): {e!r}. Devolviendo -inf.")
                        rv = float('-inf')
                    results.append(rv)
        
                elif m == 'cluster_persistence':
                    try:
                        pers_vals = getattr(estimator.model_, 'cluster_persistence_', [])
                        if len(pers_vals) == 0:
                            max_p = 0.0
                            print(f"  [CP] No hay clusters (lista vacía). max persistence → 0.0")
                        else:
                            max_p = float(max(pers_vals))
                            print(f"  [CP] Persistencia máxima entre clusters: {round(max_p, 6)}")
                        if np.isnan(max_p):
                            print(f"  [CP] max_p es NaN. Lo convierto a -inf.")
                            max_p = float('-inf')
                    except Exception as e:
                        print(f"  [CP] Trial #{trial.number}: excepción al obtener cluster_persistence_: {e!r}. Devolviendo -inf.")
                        max_p = float('-inf')
                    results.append(max_p)
                elif m == 'dbcv':
                    # 1) Recuperar etiquetas y podar según persistence_threshold
                    labels = estimator.model_.labels_.copy()
                    pers   = estimator.model_.cluster_persistence_
                    if persistence_threshold > 0.0:
                        for cid, pval in enumerate(pers):
                            if pval < persistence_threshold:
                                labels[labels == cid] = -1
                
                    # 2) Contar clusters válidos tras poda
                    unique_labels = set(labels)
                    unique_labels.discard(-1)
                    n_clusters = len(unique_labels)
                
                    # 3) Validar que hay al menos 2 clusters
                    if n_clusters < 2:
                        if n_clusters == 0:
                            print(f"    → DBCV no se puede calcular: 0 clusters (todo ruido).")
                        else:
                            only_one = next(iter(unique_labels))
                            print(f"    → DBCV no se puede calcular: solo 1 cluster ({only_one}).")
                        results.append(float('-inf'))
                    else:
                        # 4) Llamada directa a validity_index (sin pasar MST)
                        try:
                            X_arr = X  # asumimos que X ya es numpy array; si fuera DataFrame, usare `X.values`
                        except:
                            X_arr = np.asarray(X)
                
                        try:
                            # Esta llamada puede ser lenta, pero así usa validity_index “puro”
                            dbcv_score = validity_index(X_arr, labels, metric='euclidean')
                        except Exception as e:
                            print(f"    → Excepción en validity_index: {e!r}. Uso -inf.")
                            results.append(float('-inf'))
                            continue
                
                        if np.isnan(dbcv_score):
                            print(f"    → validity_index devolvió NaN. Uso -inf.")
                            results.append(float('-inf'))
                        else:
                            results.append(float(dbcv_score))
                        
                else:
                    raise ValueError(f"Unknown metric '{m}'")
        
            # 5) Comprobamos que devolvemos exactamente len(methods) valores
            if len(results) != len(methods):
                raise RuntimeError(
                    f"Trial {trial.number} devolvió {len(results)} valores, pero esperábamos {len(methods)}."
                )
        
            return tuple(results) if len(results) > 1 else results[0]

        # 4) Run the optimization
        study.optimize(objective, n_trials=n_trials)

        # 5) Select the best trial
        if len(methods) > 1:
            self._pareto_trials = study.best_trials
            best_trial = self._pareto_trials[0]
        else:
            best_trial = study.best_trial

        # 6) Retrain final estimator with best parameters
        final_params = {**hdbscan_kwargs, **best_trial.params}
        self.clusterer = HDBSCANEstimator(
            persistence_threshold=persistence_threshold,
            **final_params
        ).fit(X).model_
        self.best_params_ = best_trial.params
        self.best_score_  = (
            best_trial.values if hasattr(best_trial, 'values') else best_trial.value
        )

    def show_results(self):
        """Print best params + objectives (and Pareto front if multi-objective)."""
        if self._study is None:
            print("Run .search(...) first.")
            return

        # single best
        t = (self._pareto_trials[0] if self._pareto_trials else self._study.best_trial)
        print("Params:", t.params)

        # objectives
        if hasattr(t, 'values'):
            print("Objectives:", tuple(f"{v:.6f}" for v in t.values))
        else:
            print("Objective:", f"{t.value:.6f}")

        # Pareto front info
        if self._pareto_trials:
            print(f"Pareto front size: {len(self._pareto_trials)}")

    def get_best_params(self) -> dict | None:
        return self.best_params_

    def save_results(self, filename: str, format: str = 'csv') -> None:
        if self.combined_data is None:
            raise ValueError('No results to save; run search first')
        self.combined_data.write(filename, format=format, overwrite=True)

    def clustering_statistics(self, show_outliers: bool = False) -> None:
        table = self.combined_data or self.data
        labels = np.array(table['cluster'].data, dtype=int)
        total = labels.size
        outliers = int((labels == -1).sum())
        if not show_outliers:
            labels = labels[labels != -1]
        if labels.size == 0:
            print('No clusters found')
            return
        unique, counts = np.unique(labels, return_counts=True)
        stats = {
            'total_stars': total,
            'outliers': outliers,
            'clusters': int(len(unique)),
            'mean': float(np.mean(counts)),
            'median': float(np.median(counts)),
            'std': float(np.std(counts)),
            'min': int(np.min(counts)),
            'max': int(np.max(counts)),
        }
        print('Clustering Statistics:', stats)

    def plot_grid_search_results(self) -> None:
        """Line plot de min_cluster_size vs mean_test_score (GridSearchCV)."""
        if not self.cv_results_:
            raise ValueError("No CV results; run grid search first")
        params = np.array(self.cv_results_["param_min_cluster_size"].data, dtype=int)
        scores = np.array(self.cv_results_["mean_test_score"], dtype=float)
        valid = ~np.isnan(scores)
        fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)
        ax.plot(params[valid], scores[valid], marker='o', linestyle='-')
        ax.set(xlabel="min_cluster_size",
               ylabel="Mean Test Score",
               title="Grid Search Results")
        ax.grid(True)
        plt.show()

    def plot_pm_scatter(self, pm_columns=['pmra','pmdec'], show_outliers=False, clusters=None):
        """Scatter of proper motion colored by cluster."""
        df = self.data[pm_columns + ['cluster']].to_pandas()
        if not show_outliers:
            df = df[df['cluster'] != -1]
        if clusters:
            df = df[df['cluster'].isin(clusters)]
        fig, ax = plt.subplots(figsize=(8,6))
        cmap = plt.get_cmap('tab10')
        for i, cl in enumerate(sorted(df['cluster'].unique())):
            sub = df[df['cluster'] == cl]
            ax.scatter(sub[pm_columns[0]], sub[pm_columns[1]],
                       label=str(cl), s=40, alpha=0.8,
                       color=cmap(i % 10), edgecolor='none')
        ax.set(title='Proper Motion Scatter', xlabel=pm_columns[0], ylabel=pm_columns[1])
        ax.legend(title='Cluster', loc='upper left', bbox_to_anchor=(1.05,1))
        plt.tight_layout()
        plt.show()

    def plot_probability_histogram(self):
        """Histogram of membership probabilities."""
        probs = np.array(self.data['probability_hdbscan'].data, float)
        fig, ax = plt.subplots(figsize=(8,5))
        ax.hist(probs, bins=25, histtype='stepfilled', alpha=0.7, edgecolor='k')
        ax.set(title='Membership Probability Distribution', xlabel='Probability', ylabel='Count')
        plt.tight_layout(); plt.show()

    def plot_cluster_members(self, show_outliers=False):
        """Bar chart of members per cluster."""
        df = self.combined_data.to_pandas() if self.combined_data else self.data.to_pandas()
        if not show_outliers:
            df = df[df['cluster'] != -1]
        counts = df['cluster'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8,5))
        counts.plot.bar(ax=ax, edgecolor='k')
        ax.set(title='Cluster Member Counts', xlabel='Cluster', ylabel='Count')
        plt.tight_layout(); plt.show()

    def plot_cluster_persistence(self) -> None:
        """Bar plot de persistence score de cada cluster basado en summary."""
        summary = self.get_cluster_summary(include_noise=True)
        labels = summary['cluster'].astype(str)
        pers = summary['persistence'].values
        fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
        ax.bar(labels, pers, edgecolor='k')
        ax.set(title='Cluster Persistence', xlabel='Cluster', ylabel='Persistence')
        ax.grid(axis='y')
        plt.show()

    def plot_members_vs_persistence(self, show_outliers: bool = False) -> None:
        """Scatter: tamaño de cluster vs su persistence basado en summary."""
        summary = self.get_cluster_summary(include_noise=True)
        if not show_outliers:
            summary = summary[summary['cluster'] != -1]
        counts = summary['count'].values
        pers = summary['persistence'].values
        labels = summary['cluster'].values
        fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
        ax.scatter(counts, pers, s=80, alpha=0.8)
        for cnt, p, lbl in zip(counts, pers, labels):
            ax.annotate(str(lbl), (cnt, p), textcoords='offset points', xytext=(5, 5))
        ax.set(title='Cluster Persistence vs. Number of Members', xlabel='Number of Members', ylabel='Cluster Persistence')
        ax.grid(True)
        plt.show()

    def get_cluster_summary(
        self,
        pm_columns: Sequence[str] = ("pmra", "pmdec"),
        include_noise: bool = True,
    ) -> pd.DataFrame:
        """
        Return a pandas DataFrame summarizing each cluster.
    
        Always-produced columns:
            - cluster, count, fraction, persistence, mean_prob, median_prob,
              min_prob, max_prob, iqr_prob,
              centroid_<pm_i>, mean_dist2centroid, std_dist2centroid, <pm_i>_range.
    
        Behavior:
            - If a fitted HDBSCAN model is available (self.clusterer with
              cluster_persistence_), 'persistence' is read from there by index.
            - Otherwise (e.g., loading from an .ecsv with labels), 'persistence'
              is filled with NaN.
    
        Parameters
        ----------
        pm_columns
            Proper-motion column names to summarize. The function will always
            include centroid/range/dist columns for these names; if the columns
            are missing in the data, NaN is returned for those metrics.
        include_noise
            If False, drop noise label (-1) before summarizing.
    
        Notes
        -----
        - NaNs are handled safely in all statistics.
        - If the persistence array is shorter than the max cluster label or the
          model is absent, 'persistence' is NaN for those labels.
        """
        table = self.combined_data or self.data
        if table is None:
            raise ValueError("No data available. Run search() or assign .data first.")
    
        df = table.to_pandas()
    
        # Basic guards
        if "cluster" not in df or "probability_hdbscan" not in df:
            raise KeyError(
                "Required columns 'cluster' and 'probability_hdbscan' were not found."
            )
    
        if not include_noise:
            df = df[df["cluster"] != -1]
    
        total = len(df)
    
        # Try to read persistence from fitted model; may not exist (e.g., ECSV-only)
        pers = np.asarray(
            getattr(getattr(self, "clusterer", None), "cluster_persistence_", []),
            dtype=float,
        )
        # We will try to index pers by cluster label; if not possible → NaN.
    
        # Prepare output schema to be stable
        out_cols = [
            "cluster", "count", "fraction", "persistence",
            "mean_prob", "median_prob", "min_prob", "max_prob", "iqr_prob",
            *(f"centroid_{c}" for c in pm_columns),
            "mean_dist2centroid", "std_dist2centroid",
            *(f"{c}_range" for c in pm_columns),
        ]
        records: list[dict] = []
    
        if total == 0:
            return pd.DataFrame(columns=out_cols)
    
        # Do we have all pm columns?
        have_pm = all(c in df.columns for c in pm_columns)
    
        for lbl, group in df.groupby("cluster", sort=True):
            # Probabilities (robust to NaNs)
            p = group["probability_hdbscan"].astype(float).to_numpy()
            p = p[np.isfinite(p)]
            if p.size == 0:
                p_mean = p_med = p_min = p_max = p_iqr = np.nan
            else:
                p_mean = float(np.nanmean(p))
                p_med = float(np.nanmedian(p))
                p_min = float(np.nanmin(p))
                p_max = float(np.nanmax(p))
                p_iqr = float(iqr(p, nan_policy="omit"))
    
            # Persistence from model if available; else NaN
            if pers.size > 0 and isinstance(lbl, (int, np.integer)) and 0 <= int(lbl) < pers.size:
                pval = float(pers[int(lbl)])
            else:
                pval = np.nan
    
            rec = {
                "cluster": lbl,
                "count": int(len(group)),
                "fraction": float(len(group) / total) if total > 0 else 0.0,
                "persistence": pval,
                "mean_prob": p_mean,
                "median_prob": p_med,
                "min_prob": p_min,
                "max_prob": p_max,
                "iqr_prob": p_iqr,
            }
    
            # Kinematics (always include the columns in the schema; fill NaN if missing)
            if have_pm:
                pts = group[list(pm_columns)].astype(float).to_numpy()
                finite_rows = np.all(np.isfinite(pts), axis=1)
                if np.any(finite_rows):
                    pts_f = pts[finite_rows]
                    centroid = np.nanmean(pts_f, axis=0)
                    dists = np.linalg.norm(pts_f - centroid, axis=1)
                    rec.update({
                        **{f"centroid_{pm_columns[i]}": float(centroid[i])
                           for i in range(len(pm_columns))},
                        "mean_dist2centroid": float(np.nanmean(dists)),
                        "std_dist2centroid": float(np.nanstd(dists)),
                        **{f"{pm_columns[i]}_range": float(
                               np.nanmax(pts_f[:, i]) - np.nanmin(pts_f[:, i])
                           ) for i in range(len(pm_columns))}
                    })
                else:
                    rec.update({
                        **{f"centroid_{c}": np.nan for c in pm_columns},
                        "mean_dist2centroid": np.nan,
                        "std_dist2centroid": np.nan,
                        **{f"{c}_range": np.nan for c in pm_columns},
                    })
            else:
                rec.update({
                    **{f"centroid_{c}": np.nan for c in pm_columns},
                    "mean_dist2centroid": np.nan,
                    "std_dist2centroid": np.nan,
                    **{f"{c}_range": np.nan for c in pm_columns},
                })
    
            records.append(rec)
    
        return (
            pd.DataFrame.from_records(records, columns=out_cols)
            .sort_values("cluster")
            .reset_index(drop=True)
        )
