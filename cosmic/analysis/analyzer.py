"""High-level cluster analysis utilities."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
from astropy.table import QTable

from clustering import Clustering
from ._clipping import sigma_clip_parallax as _sigma_clip_parallax
from ._io import load_dataset
from ._plots import (
    plot_persistence_vs_members,
    plot_probability_vs_gmag,
    plot_pms_panels,
)
from ._sagitta import pms_characterization as _pms_characterization


class ClusterAnalyzer:
    """High-level helper around :class:`~clustering.Clustering` and plotting utilities."""

    def __init__(
        self,
        file_obj,
        *,
        dataloader_kwargs: dict | None = None,
        dill_cache: bool = True,
        search_method: str = 'optuna',
        output_dir: str | None = None,
        verbose: int = logging.INFO,
        debug_mode: bool = False,
    ) -> None:
        self.selected_cluster: int | None = None
        self.search_method = search_method

        result = load_dataset(
            file_obj,
            dataloader_kwargs=dataloader_kwargs,
            dill_cache=dill_cache,
            output_dir=output_dir,
            verbose=verbose,
            debug_mode=debug_mode,
        )

        self.data: QTable = result['data']
        self.loader = result['loader']
        self.clustering: Clustering | None = result['clustering']
        self.base_dir = Path(result['base_dir'])
        self.output_dir = Path(result['output_dir'])
        self.source = result['source']
        self.dill_path = result['dill_path']

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------
    def clusters_summary(self, **kwargs) -> pd.DataFrame:
        if self.clustering is not None:
            return self.clustering.get_cluster_summary(**kwargs)
        tmp = Clustering(self.data, search_method=self.search_method)
        return tmp.get_cluster_summary(**kwargs)

    # ------------------------------------------------------------------
    # Plotting wrappers
    # ------------------------------------------------------------------
    def plot_persistence_vs_members(
        self,
        *,
        percentile: float = 0.8,
        figsize: tuple[int, int] = (10, 10),
        save: bool = False,
        filename: str | None = None,
        **plot_kwargs,
    ) -> None:
        summary = self.clusters_summary()
        plot_persistence_vs_members(
            self.data,
            summary,
            percentile=percentile,
            output_dir=self.output_dir,
            save=save,
            filename=filename,
            figsize=figsize,
            plot_kwargs=plot_kwargs,
        )

    def plot_probability_vs_gmag(
        self,
        *,
        cluster: int | None = None,
        prob_thresh: float = 0.6,
        gmag_limit: float | None = None,
        figsize: tuple[int, int] = (7, 6),
        save: bool = False,
        filename: str | None = None,
        **plot_kwargs,
    ) -> None:
        cid = cluster if cluster is not None else self.selected_cluster
        if cid is None:
            raise ValueError('No cluster specified; call select_cluster() first or pass cluster argument.')
        self.selected_cluster = cid
        plot_probability_vs_gmag(
            self.data,
            cluster=cid,
            probability_threshold=prob_thresh,
            gmag_limit=gmag_limit,
            figsize=figsize,
            output_dir=self.output_dir,
            save=save,
            filename=filename,
            plot_kwargs=plot_kwargs,
        )

    def plot_pms(
        self,
        *,
        cluster: int | None = None,
        pms_threshold: float = 0.6,
        figsize: tuple[int, int] = (7, 7),
        layout: str = 'tight',
        save: bool = False,
        filename: str | None = None,
        age_xlim: tuple[float, float] | None = None,
        av_xlim: tuple[float, float] | None = None,
    ) -> None:
        cid = cluster if cluster is not None else self.selected_cluster
        if cid is None:
            raise ValueError('No cluster specified; call select_cluster() first or pass `cluster`.')
        self.selected_cluster = cid
        plot_pms_panels(
            self.data,
            cluster=cid,
            pms_threshold=pms_threshold,
            figsize=figsize,
            layout=layout,
            output_dir=self.output_dir,
            save=save,
            filename=filename,
            age_xlim=age_xlim,
            av_xlim=av_xlim,
        )

    # ------------------------------------------------------------------
    # Data manipulation helpers
    # ------------------------------------------------------------------
    def select_cluster(self, label: int):
        self.selected_cluster = label
        return self.data[self.data['cluster'] == label]

    def sigma_clip_parallax(
        self,
        *,
        cluster: int | None = None,
        sigma: float = 2.0,
        print_results: bool = False,
        use_biweight: bool = True,
        in_place: bool = True,
        mark_label: int = -1,
        return_mask: bool = False,
        preselector_mask=None,
    ):
        cid = cluster if cluster is not None else self.selected_cluster
        if cid is None:
            raise ValueError('No cluster specified; call select_cluster() first or pass cluster argument.')
        result = _sigma_clip_parallax(
            self.data,
            cluster=cid,
            sigma=sigma,
            use_biweight=use_biweight,
            in_place=in_place,
            mark_label=mark_label,
            print_results=print_results,
            return_mask=return_mask,
            preselector_mask=preselector_mask,
        )
        return result

    def pms_characterization(
        self,
        *,
        cluster: int | None = None,
        run_cli: bool = False,
        input_dir: str | None = None,
        output_prefix: str | None = None,
        overwrite_inputs: bool = True,
        return_data: bool = False,
    ):
        cid = cluster if cluster is not None else self.selected_cluster
        if cid is None:
            raise ValueError('No cluster specified; call select_cluster() first or pass `cluster` argument.')
        joined = _pms_characterization(
            self.data,
            cluster=cid,
            base_dir=self.base_dir,
            input_dir=input_dir,
            output_prefix=output_prefix,
            overwrite_inputs=overwrite_inputs,
            run_cli=run_cli,
            return_data=return_data,
        )
        return joined


__all__ = ['ClusterAnalyzer']
