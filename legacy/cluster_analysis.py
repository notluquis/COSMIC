import os
import dill
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from adjustText import adjust_text
from astropy.table import QTable
from data_loader import DataLoader
from clustering import Clustering
from matplotlib import ticker
import astropy.units as u
from astropy.stats import sigma_clip, biweight_location, biweight_scale, mad_std
from collections import Counter
import warnings
import json
import datetime as dt
import arviz as az
import pymc as pm
import pytensor
import pytensor.tensor as pt
import logging

import os, dill, logging
import pandas as pd
from astropy.table import QTable, Table
from data_loader import DataLoader
from clustering import Clustering

from contextlib import contextmanager

@contextmanager
def _optuna_safe_unpickle():
    """Evita que Optuna intente abrir el SQLite durante dill.load()."""
    try:
        from optuna.storages._rdb.storage import RDBStorage
        _orig = RDBStorage.__setstate__
        def _patched(self, state):
            # Restaurar atributos sin crear engine/DB
            for k, v in state.items():
                setattr(self, k, v)
            # anular conexiones
            for attr in ("engine", "_engine", "scoped_session"):
                if hasattr(self, attr):
                    setattr(self, attr, None)
        RDBStorage.__setstate__ = _patched
        yield
    except Exception:
        # Si no existe esa ruta en tu versión de Optuna, seguimos sin parche
        yield
    finally:
        try:
            RDBStorage.__setstate__ = _orig
        except Exception:
            pass

class ClusterAnalyzer:
    def __init__(self,
                 file_obj,
                 *,
                dataloader_kwargs=None,
                 dill_cache: bool = True,
                 search_method: str = 'optuna',   # se conserva la firma
                 output_dir: str | None = None,
                 verbose: int = logging.INFO,
                 debug_mode: bool = False):
        """
        Inicializa el analizador:
        - Si `file_obj` es ruta: puede ser .dill (con Clustering) o .ecsv/.fits/etc.
        - Si `file_obj` es DataFrame/QTable/Table: lo usa en memoria.
        - Si hay .dill hermano (mismo stem), se usa como caché preferente.
        """
        self.data = None
        self.loader = None
        self.clustering = None
        self.selected_cluster = None

        dataloader_kwargs = dataloader_kwargs or {}

        # --- Caso 1: file_obj es ruta ---
        if isinstance(file_obj, str):
            file_path = os.path.abspath(file_obj)
            base, ext = os.path.splitext(file_path)
            self.base_dir = os.path.dirname(file_path) or os.getcwd()
            self.source = file_path
            self.dill_path = base + ".dill"

            # Output dir
            self.output_dir = output_dir or self.base_dir
            os.makedirs(self.output_dir, exist_ok=True)

            is_input_dill = ext.lower() == ".dill"
            dill_to_open = None

            # Si el input ya es .dill → úsalo
            if is_input_dill and os.path.exists(file_path):
                dill_to_open = file_path

            if dill_to_open is not None:
                # ---------- Cargar .dill ----------
                try:
                    with open(dill_to_open, "rb") as f:
                        payload = dill.load(f)
                except Exception:
                    # si falló por SQLite, reintenta con parche
                    with _optuna_safe_unpickle():
                        with open(dill_to_open, "rb") as f:
                            payload = dill.load(f)

                if isinstance(payload, Clustering):
                    # caché con objeto Clustering completo (tu caso)
                    self.clustering = payload
                    self.data = payload.data

                elif isinstance(payload, dict):
                    # caché estandarizada {"data": QTable, ...}
                    if "data" in payload:
                        self.data = payload["data"]
                    elif "table" in payload:
                        self.data = payload["table"]
                    elif "qtable" in payload:
                        self.data = payload["qtable"]
                    else:
                        raise KeyError(
                            f"El .dill '{dill_to_open}' no tiene claves conocidas ('data','table','qtable')."
                        )
                elif isinstance(payload, (QTable, Table)):
                    self.data = payload if isinstance(payload, QTable) else QTable(payload)

                elif isinstance(payload, pd.DataFrame):
                    self.data = QTable.from_pandas(payload)

                else:
                    raise TypeError(
                        f"Contenido del .dill no soportado: {type(payload)}. "
                        "Esperaba Clustering, dict, QTable/Table o pandas.DataFrame."
                    )

                # Normaliza usando DataLoader “ligero”
                self.loader = DataLoader(self.source, verbose=verbose, debug_mode=debug_mode)
                self.loader.data = self.data
                self.loader._handle_masked_data()
                self.loader.conv_wrong_units()
                self.data = self.loader.data
                return

            # ---------- No hay .dill aplicable: leer archivo con DataLoader ----------
            self.loader = DataLoader(file_path, verbose=verbose, debug_mode=debug_mode)
            self.data = self.loader.load_data(**dataloader_kwargs)
            # guarda caché estandarizada si procede
            if dill_cache and not is_input_dill:
                payload_to_save = {
                    "data": self.data,
                    "source": file_path,
                    "loader_kwargs": dataloader_kwargs,
                }
                with open(self.dill_path, "wb") as f:
                    dill.dump(payload_to_save, f, protocol=dill.HIGHEST_PROTOCOL)
            return

        # --- Caso 2: DataFrame / QTable / Table en memoria ---
        if isinstance(file_obj, pd.DataFrame):
            qtab = QTable.from_pandas(file_obj)
            print('df')
        elif isinstance(file_obj, (QTable, Table)):
            qtab = file_obj if isinstance(file_obj, QTable) else QTable(file_obj)
            print('qtable')
        else:
            raise TypeError("file_obj debe ser ruta str, pandas.DataFrame o astropy Table/QTable.")
            print('else')

        self.source = "<in-memory>"
        self.base_dir = os.getcwd()
        self.output_dir = output_dir or self.base_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Normaliza con DataLoader
        self.loader = DataLoader(self.source, verbose=verbose, debug_mode=debug_mode)
        self.loader.data = qtab
        self.loader._handle_masked_data()
        self.loader.conv_wrong_units()
        self.data = self.loader.data
                
    def clusters_summary(self, **kwargs) -> pd.DataFrame:
        if getattr(self, "clustering", None) is not None:
            return self.clustering.get_cluster_summary(**kwargs)
        # si no existe self.clustering pero tienes self.data con columnas,
        # crea un Clustering “temporal” sin modelo si quieres, o bien llama
        # a la misma lógica aquí; lo más limpio es delegar siempre al método
        # anterior cuando haya un objeto Clustering disponible.
        from clustering import Clustering
        tmp = Clustering(self.data)  # sin modelo
        return tmp.get_cluster_summary(**kwargs)

    def plot_persistence_vs_members(self,
                                    percentile: float = 0.8,
                                    figsize: tuple[int,int] = (10,10),
                                    save: bool = False,
                                    filename: str | None = None,
                                    **plot_kwargs):
        """
        Plot Persistence vs Number of Members with:
          - marginal histograms,
          - labels with arrows,
          - color by standard deviation of total PM.
        """
        # compute per-cluster PM-std
        df_all = self.data.to_pandas()
        pm_std = df_all.groupby('cluster')['pm'].std()
        summary = self.clusters_summary()
        summary['pm_std'] = summary['cluster'].map(pm_std)

        # pick top percentile
        thr = summary['persistence'].quantile(percentile)
        df = summary.query("cluster != -1")
        df_top = df.query("persistence >= @thr")

        # build colormap
        norm = plt.Normalize(df_top.pm_std.min(), df_top.pm_std.max())
        cmap = plt.colormaps['plasma']
        colors = cmap(norm(df_top.pm_std))

        # setup axes + marginals
        fig, ax = plt.subplots(figsize=figsize)
        div = make_axes_locatable(ax)
        ax_top = div.append_axes("top", size="20%", pad=0, sharex=ax)
        ax_rt = div.append_axes("right", size="20%", pad=0, sharey=ax)
        for a in (ax, ax_top, ax_rt):
            a.grid(False)

        # persistence histogram
        ax_top.hist(df.persistence, bins=plot_kwargs.get('bins','auto'),
                    color='lightgray', edgecolor='k')
        ax_top.axvline(thr, color='C0', linestyle='--')
        ax_top.xaxis.set_visible(False)
        ax_top.spines['bottom'].set_visible(False)

        # size histogram
        ax_rt.hist(df['count'], bins=plot_kwargs.get('bins','auto'),
                   orientation='horizontal', color='lightgray', edgecolor='k')
        ax_rt.axhline(df['count'].quantile(percentile), color='C0', linestyle='--')
        ax_rt.yaxis.set_visible(False)
        ax_rt.spines['left'].set_visible(False)

        # main scatter
        ax.scatter(df.persistence, df['count'],
                   c='lightgray', s=plot_kwargs.get('s_all',40),
                   alpha=plot_kwargs.get('alpha_all',0.5), edgecolor='k')
        ax.scatter(df_top.persistence, df_top['count'],
                   c=colors, s=plot_kwargs.get('s_top',80),
                   alpha=plot_kwargs.get('alpha_top',0.9), edgecolor='k')

        # labels + repel
        texts = [ax.text(x, y, str(c), ha='center', va='center',
                         zorder=5, fontsize=plot_kwargs.get('fontsize',11))
                 for x, y, c in zip(df_top.persistence, df_top['count'], df_top.cluster)]
        _, arrows = adjust_text(
            texts,
            x=df.persistence, y=df['count'],
            ax=ax,
            expand=plot_kwargs.get('expand',(3.4,3.4)),
            force_points=plot_kwargs.get('force_points',0.4),
            arrowprops=plot_kwargs.get('arrowprops',dict(
                arrowstyle='->', color='gray', lw=0.8,
                shrinkA=plot_kwargs.get('shrinkA',10),
                shrinkB=plot_kwargs.get('shrinkB',10),
                connectionstyle=plot_kwargs.get('connectionstyle',
                    'angle3,angleA=90,angleB=0'),
                zorder=1
            )),
            return_objects=True
        )
        for arr in arrows:
            arr.set_alpha(plot_kwargs.get('arrow_alpha',0.7))

        # colorbar under scatter
        sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax,
                            orientation='horizontal',
                            fraction=plot_kwargs.get('cbar_fraction',0.046),
                            pad=plot_kwargs.get('cbar_pad',0.065))
        cbar.set_label(plot_kwargs.get('cbar_label','Std Dev of Total PM'),
                       fontsize=plot_kwargs.get('cbar_fontsize',12))

        ax.set(xlabel=plot_kwargs.get('xlabel','Persistence'),
               ylabel=plot_kwargs.get('ylabel','Number of Members'))

        # save figure if requested
        if save:
            fname = filename or 'persistence_vs_members.png'
            fig.savefig(os.path.join(self.output_dir, fname),
                        dpi=plot_kwargs.get('dpi',300), bbox_inches='tight')
        plt.show()
    def plot_probability_vs_gmag(self,
                                 cluster: int | None = None,
                                 prob_thresh: float = 0.6,
                                 gmag_limit: float | None = None,
                                 figsize: tuple[int,int] = (7,6),
                                 save: bool = False,
                                 filename: str | None = None,
                                 **plot_kwargs):
        """
        Plot Gmag vs HDBSCAN probability colored by fidelity, with thresholds.
        If cluster is None, uses last selected via select_cluster().
        """
        # determine cluster
        cid = cluster if cluster is not None else self.selected_cluster
        if cid is None:
            raise ValueError("No cluster specified; call select_cluster() first or pass cluster argument.")

        self.selected_cluster = cid
        df = self.data.to_pandas()
        sub = df[df['cluster'] == cid]
        if gmag_limit is not None:
            sub = sub[sub['Gmag'] <= gmag_limit]

        fig, ax = plt.subplots(figsize=figsize)
        sc = ax.scatter(sub['Gmag'], sub['probability_hdbscan'],
                        s=plot_kwargs.get('s',9),
                        c=sub['fidelity_v2'], cmap=plot_kwargs.get('cmap','coolwarm'),
                        vmin=sub['fidelity_v2'].min(), vmax=sub['fidelity_v2'].max())
        cbar = fig.colorbar(sc, ax=ax, pad=plot_kwargs.get('cbar_pad',0.01))
        cbar.set_label(plot_kwargs.get('cbar_label','Astrometric fidelity'),
                       fontsize=plot_kwargs.get('cbar_fontsize',15))
        cbar.ax.tick_params(labelsize=plot_kwargs.get('cbar_tick_size',14))

        ax.axhline(prob_thresh, color='k', ls='--',
                   label=plot_kwargs.get('prob_label',f'{int(prob_thresh*100)}% prob'))
        ax.set_xlabel(plot_kwargs.get('xlabel',f"Gmag [{self.data['Gmag'].unit} ]"),
                      fontsize=plot_kwargs.get('xlabel_size',16))
        ax.set_ylabel(plot_kwargs.get('ylabel','probability_hdbscan'),
                      fontsize=plot_kwargs.get('ylabel_size',16))
        ax.legend(loc=plot_kwargs.get('legend_loc','lower left'),
                  fontsize=plot_kwargs.get('legend_size',10),
                  framealpha=plot_kwargs.get('legend_alpha',0.4))

        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(axis='both', which='both',
                       direction=plot_kwargs.get('tick_direction','in'),
                       labelsize=plot_kwargs.get('tick_labelsize',14))

        fig.align_labels()
        if save:
            fname = filename or f'prob_vs_gmag_cluster{cid}.pdf'
            fig.savefig(os.path.join(self.output_dir, fname),
                        dpi=plot_kwargs.get('dpi','figure'), bbox_inches='tight')
        plt.show()
    def select_cluster(self, label: int):
        """Select a cluster and return its Astropy table."""
        self.selected_cluster = label
        return self.data[self.data['cluster'] == label]

    def sigma_clip_parallax(
        self,
        cluster: int | None = None,
        sigma: float = 2.0,
        print_results: bool = False,
        use_biweight: bool = True,
        in_place: bool = True,
        mark_label: int = -1,
        return_mask: bool = False,
        preselector_mask=None,  # optional boolean mask over full table
    ):
        """
        Apply robust sigma clipping on 'parallax' for the given cluster (or last selected).
    
        If `preselector_mask` is provided, bounds are computed on rows that are BOTH in the
        target cluster AND in the preselector. After computing bounds, within the chosen cluster:
          - ONLY rows that are in (preselector_mask AND within [lower, upper]) remain as members,
          - ALL OTHER rows of that cluster are relabeled with `mark_label` (default -1).
    
        Returns
        -------
        If in_place:
            (lower, upper) or (lower, upper, inliers_mask_full, outliers_mask_full) if return_mask=True
        Else:
            (lower, upper, data_copy) or (lower, upper, data_copy, inliers_mask_full, outliers_mask_full)
        """
        # 1) pick cluster
        cid = cluster if cluster is not None else self.selected_cluster
        if cid is None:
            raise ValueError("No cluster specified; call select_cluster() first or pass cluster argument.")
    
        clu_col = self.data['cluster']
        par_col = self.data['parallax']
    
        # 2) masks
        base_mask = (clu_col == cid)  # all rows of the chosen cluster
    
        # selection used to compute bounds
        if preselector_mask is not None:
            if getattr(preselector_mask, "shape", None) != base_mask.shape:
                raise ValueError("preselector_mask must have the same length as the table.")
            if hasattr(preselector_mask, "dtype") and preselector_mask.dtype is not bool:
                preselector_mask = np.asarray(preselector_mask, dtype=bool)
            selection_mask = base_mask & preselector_mask
        else:
            selection_mask = base_mask
    
        # 3) pick parallax to clip
        plx = par_col[selection_mask]
    
        # guards
        if getattr(plx, "size", len(plx)) == 0:
            raise ValueError("No rows selected for sigma clipping (mask is empty).")
        n_finite = np.count_nonzero(np.isfinite(plx))
        if n_finite < 3:
            raise ValueError(f"Not enough finite parallax values for sigma clipping: {n_finite}")
    
        # 4) robust estimators
        cenfunc = biweight_location
        stdfunc = biweight_scale if use_biweight else mad_std
    
        # 5) sigma clip (NaNs ignored)
        _, lower, upper = sigma_clip(
            plx,
            sigma=sigma,
            cenfunc=cenfunc,
            stdfunc=stdfunc,
            return_bounds=True,
        )
    
        if not (np.isfinite(lower) and np.isfinite(upper)):
            raise RuntimeError(f"sigma_clip returned non-finite bounds: lower={lower}, upper={upper}")
    
        # 6) final membership logic (ONLY keep: in preselector AND within bounds)
        within_bounds = (par_col >= lower) & (par_col <= upper)
        if preselector_mask is not None:
            final_keep = base_mask & preselector_mask & within_bounds
        else:
            final_keep = base_mask & within_bounds
    
        to_noise = base_mask & ~final_keep  # everything else in this cluster → mark_label
    
        # 7) apply or return
        if in_place:
            # Optional safety: if writing a negative mark_label, ensure signed int
            if mark_label < 0:
                assert np.issubdtype(self.data['cluster'].dtype, np.signedinteger), \
                    f"'cluster' dtype must be signed int to assign {mark_label}, got {self.data['cluster'].dtype}"
    
            self.data['cluster'][to_noise] = mark_label
    
            if print_results:
                print(f"{int(base_mask.sum())} before sigma clipping")
                print(f"{int(final_keep.sum())} after sigma clipping")
    
            if return_mask:
                return lower, upper, final_keep, to_noise
            return lower, upper
        else:
            data_copy = self.data.copy(copy_data=True)
            data_copy['cluster'][to_noise] = mark_label
    
            if print_results:
                print(f"{int(base_mask.sum())} before sigma clipping")
                print(f"{int(final_keep.sum())} after sigma clipping")
    
            if return_mask:
                return lower, upper, data_copy, final_keep, to_noise
            return lower, upper, data_copy
    def pms_characterization(self,
                             cluster: int | None = None,
                             run_cli: bool = False,
                             input_dir: str | None = None,
                             output_prefix: str | None = None,
                             overwrite_inputs: bool = True,
                             return_data: bool = False,
        ):
        """
            Build Sagitta input for the selected cluster, (optionally) run the Sagitta CLI,
            read its outputs (Av, PMS prob, age), and merge them back into `self.data`
            for the chosen cluster (joined on `source_id`).
        
            Parameters
            ----------
            cluster : int | None
                Cluster label to process. If None, uses `self.selected_cluster`.
            run_cli : bool
                If True, runs the external `sagitta` command on the generated FITS.
                Requires `sagitta` to be available in PATH.
            input_dir : str
                Directory to write/read the Sagitta input/output FITS files.
            output_prefix : str | None
                Base name for the input FITS. If None, uses `cluster_{cid}_sagitta`.
                The Sagitta output is expected as `{output_prefix}-sagitta.fits`.
            overwrite_inputs : bool
                Overwrite the input FITS if it already exists.
            return_data : bool
                If True, also return the updated Astropy table for this cluster.
        
            Returns
            -------
            summary : dict
                A small dictionary with counts and basic stats.
            (optional) updated_cluster_table : QTable
                Only if `return_data=True`.
        """
        import warnings
        from astropy.table import join
        from astropy.coordinates import SkyCoord
        from pathlib import Path
        import subprocess
        cid = cluster if cluster is not None else getattr(self, "selected_cluster", None)
        if cid is None:
            raise ValueError("No cluster specified; call select_cluster() first or pass `cluster` argument.")
        
        T = self.data
        mask_cluster = (T["cluster"] == cid)
        if mask_cluster.sum() == 0:
            raise ValueError(f"Cluster {cid} has zero rows in `self.data`.")
    
        cluster_T = T[mask_cluster].copy()
    
        # --- 2) Build Sagitta input table
        # Columns Sagitta expects per tu flujo anterior:
        needed = [
            "source_id", "parallax", "Gmag", "G_BPmag", "G_RPmag", "j_m", "h_m", "ks_m",
            "parallax_error", "e_Gmag", "e_G_BPmag", "e_G_RPmag", "j_msigcom", "h_msigcom", "ks_msigcom"
        ]
        # --- coords galácticas l,b
        have_l = "l" in cluster_T.colnames
        have_b = "b" in cluster_T.colnames
        
        if not (have_l and have_b):
            if ("ra" not in cluster_T.colnames) or ("dec" not in cluster_T.colnames):
                raise ValueError("Cannot compute galactic coordinates: need 'ra' and 'dec' or preexisting 'l','b'.")
            from astropy.coordinates import SkyCoord
            sc = SkyCoord(ra=cluster_T["ra"], dec=cluster_T["dec"], unit="deg", frame="icrs")
            cluster_T["l"] = sc.galactic.l.to(u.deg)
            cluster_T["b"] = sc.galactic.b.to(u.deg)
        else:
            # Si ya existen, intenta normalizar a deg (por si vienen con unidades)
            try:
                cluster_T["l"] = cluster_T["l"].to(u.deg)
                cluster_T["b"] = cluster_T["b"].to(u.deg)
            except Exception:
                # Si no tienen unidades, las dejamos tal cual (se asumen en deg)
                pass
        # Prepare a working copy with all possible columns; fill missing with NaN
        work = QTable()
        # Mandatory ID + sky coords
        work["source_id"] = cluster_T["source_id"]
        work["parallax"]  = cluster_T["parallax"]
        work["l"]         = cluster_T["l"]
        work["b"]         = cluster_T["b"]
        # Photometry
        for c in ["Gmag", "G_BPmag", "G_RPmag", "j_m", "h_m", "ks_m"]:
            if c in cluster_T.colnames:
                work[c] = cluster_T[c]
            else:
                work[c] = np.full(len(cluster_T), np.nan)
                warnings.warn(f"[pms_characterization] Missing column '{c}', filled with NaN.")
    
        # Uncertainties: try to use provided columns; if missing, fill NaN
        for c in ["parallax_error", "e_Gmag", "e_G_BPmag", "e_G_RPmag", "j_msigcom", "h_msigcom", "ks_msigcom"]:
            if c in cluster_T.colnames:
                work[c] = cluster_T[c]
            else:
                work[c] = np.full(len(cluster_T), np.nan)
                warnings.warn(f"[pms_characterization] Missing column '{c}', filled with NaN.")
    
        # Rename to Sagitta field names
        sagitta = work["source_id","parallax","l","b",
                       "Gmag","G_BPmag","G_RPmag","j_m","h_m","ks_m",
                       "parallax_error","e_Gmag","e_G_BPmag","e_G_RPmag",
                       "j_msigcom","h_msigcom","ks_msigcom"].copy()
    
        sagitta.rename_columns(
            sagitta.colnames,
            ["source_id","parallax","l","b",
             "g","bp","rp","j","h","k",
             "eparallax","eg","ebp","erp",
             "ej","eh","ek"]
        )
    
        # --- 3) Write input FITS
        out_dir = Path(input_dir)/"sagitta" if input_dir is not None else Path(self.base_dir)/"sagitta"
        out_dir.mkdir(parents=True, exist_ok=True)
        base = output_prefix or f"cluster_{cid}_sagitta"
        in_fits  = out_dir / f"{base}.fits"
        # usa el sufijo estándar del CLI
        out_fits = out_dir / f"{base}-sagitta.fits"
    
        sagitta.write(in_fits, overwrite=overwrite_inputs, format="fits")

        # --- 4) (Optional) Run Sagitta CLI
        if run_cli:
            cmd = [
                "sagitta",
                str(in_fits),
                "--tableOut", str(out_fits),      # <<<< fuerza ruta y nombre de salida
                "--av_out", "av_sagitta",
                "--pms_out", "pms_sagitta",
                "--age_out", "age",
                # incertidumbres (50 muestreos)
                "--av_uncertainty",  "50",
                "--pms_uncertainty", "50",
                "--age_uncertainty", "50",
                "--av_scatter_range","0.1",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(
                    f"Sagitta CLI failed (code {res.returncode}).\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
                )
 
        if not out_fits.exists():
            raise FileNotFoundError(
                f"Expected output FITS not found: {out_fits}.\n"
                "If you didn't run the CLI here, ensure you put the Sagitta output there."
            )
        pms_tab = QTable.read(out_fits)
        # Ensure columns exist
        for col in ["av_sagitta", "pms_sagitta", "age"]:
            if col not in pms_tab.colnames:
                raise KeyError(f"Column '{col}' not found in {out_fits}.")
            # squeeze
            pms_tab[col] = np.squeeze(pms_tab[col])
    
        # --- 6) Join back to the cluster table and then into self.data
        joined = join(
            cluster_T,  # left
            pms_tab["source_id","av_sagitta","pms_sagitta","age"],  # right
            keys="source_id",
            join_type="left"
        )
    
        # For plotting/logic convenience: fill NaN PMS prob with 0
        if "pms_sagitta" in joined.colnames:
            joined["pms_sagitta"] = np.nan_to_num(joined["pms_sagitta"], nan=0.0)
    
        # --- 7) Write back ONLY for rows in the chosen cluster
        # Ensure columns exist in self.data
        for col in ["av_sagitta", "pms_sagitta", "age"]:
            if col not in self.data.colnames:
                self.data[col] = np.full(len(self.data), np.nan)
    
        # Build a mapping source_id -> (av, pms, age) for the cluster
        sid_to_vals = {
            int(sid): (av, pmsv, ag)
            for sid, av, pmsv, ag in zip(joined["source_id"], joined["av_sagitta"], joined["pms_sagitta"], joined["age"])
        }
    
        # Apply to self.data cluster rows
        for i, is_cl in enumerate(mask_cluster):
            if is_cl:
                sid = int(self.data["source_id"][i])
                av, pmsv, ag = sid_to_vals.get(sid, (np.nan, np.nan, np.nan))
                self.data["av_sagitta"][i]  = av
                self.data["pms_sagitta"][i] = pmsv
                self.data["age"][i]         = ag
    
        return joined if return_data else None
    def plot_pms(self,
                 cluster: int | None = None,
                 pms_threshold: float = 0.6,
                 figsize: tuple[int,int] = (7,7),
                 layout: str = "tight",
                 save: bool = False,
                 filename: str | None = None,
                 age_xlim: tuple[float,float] | None = None,
                 av_xlim: tuple[float,float] | None = None
                ):
        """
        Plot PMS diagnostics (3 stacked histograms) for the selected cluster:
          (1) PMS probability (pms_sagitta)
          (2) log(age) from Sagitta
          (3) A_V from Sagitta
    
        Curvas mostradas:
          - All data  (gris)
          - PMS       (>= pms_threshold y con 2MASS disponible)
          - Non-PMS   (<  pms_threshold y con 2MASS disponible)
          - No 2MASS  (sin fotometría 2MASS útil)
    
        Parameters
        ----------
        cluster : int | None
            Cluster label. Si None, usa self.selected_cluster.
        pms_threshold : float
            Umbral para clasificar PMS vs Non-PMS.
        figsize : (int,int)
            Tamaño de la figura.
        layout : str
            Argumento de `plt.subplots` (e.g. "tight").
        save : bool
            Si True, guarda la figura en self.output_dir.
        filename : str | None
            Nombre del archivo a guardar (por defecto 'pms_stats.pdf').
        age_xlim : tuple[float,float] | None
            Límites del eje X para log(age). Si None, se ajusta automáticamente.
        av_xlim : tuple[float,float] | None
            Límites del eje X para A_V. Si None, se ajusta automáticamente.
        """
    
        # --- 1) cluster
        cid = cluster if cluster is not None else getattr(self, "selected_cluster", None)
        if cid is None:
            raise ValueError("No cluster specified; call select_cluster() first or pass `cluster`.")
        sub = self.data[self.data["cluster"] == cid]
        if len(sub) == 0:
            raise ValueError(f"Cluster {cid} has zero rows.")
    
        # --- 2) sanity de columnas requeridas
        needed = ["pms_sagitta", "age", "av_sagitta"]
        missing = [c for c in needed if c not in sub.colnames]
        if missing:
            raise KeyError(
                "Missing columns in self.data for plotting PMS panels: "
                + ", ".join(missing)
                + ". Run `pms_characterization(...)` first."
            )
    
        # Punteros numpy para facilidad
        p_pms  = np.array(sub["pms_sagitta"])
        p_age  = np.array(sub["age"])
        p_av   = np.array(sub["av_sagitta"])
    
        # Definir “2MASS disponible”: usa presencia de Ks (o J/H/K) finito
        has_2mass = np.isfinite(np.array(sub["ks_m"])) if "ks_m" in sub.colnames else (
            (np.isfinite(np.array(sub["j_m"])) if "j_m" in sub.colnames else 0) |
            (np.isfinite(np.array(sub["h_m"])) if "h_m" in sub.colnames else 0)
        )
    
        # --- 3) máscaras de categorías
        finite_pms = np.isfinite(p_pms)
        pms_mask   = (p_pms >= pms_threshold) & has_2mass & finite_pms
        nopms_mask = (p_pms <  pms_threshold) & has_2mass & finite_pms
        no2m_mask  = ~has_2mass & finite_pms
    
        # --- 4) bins
        # pms prob [0,1]
        bins_pms = np.arange(0.0, 1.0 + 0.1, 0.1)
    
        # age bins (seguro ante NaNs o vacíos)
        finite_age = np.isfinite(p_age)
        if finite_age.any():
            a_min, a_max = np.floor(np.nanmin(p_age[finite_age])), np.ceil(np.nanmax(p_age[finite_age]))
            bins_age = np.arange(a_min, a_max + 0.1, 0.1)
        else:
            # fallback
            bins_age = np.arange(6.0, 8.1, 0.1)
    
        # A_V bins
        finite_av = np.isfinite(p_av)
        if finite_av.any():
            v_min, v_max = np.floor(np.nanmin(p_av[finite_av])), np.ceil(np.nanmax(p_av[finite_av]))
            bins_av = np.arange(v_min, v_max + 0.1, 0.1)
        else:
            bins_av = np.arange(0.0, 3.1, 0.1)
    
        # --- 5) figura
        fig, ax = plt.subplots(3, 1, layout=layout, figsize=figsize)
    
        # (1) PMS probability
        ax[0].hist(p_pms[finite_pms], bins=bins_pms, color="gray", histtype="step",
                   label="All Data", alpha=0.85)
        ax[0].hist(p_pms[pms_mask],   bins=bins_pms, color="orange", histtype="step",
                   label="PMS", alpha=0.9, linestyle='-')
        ax[0].hist(p_pms[nopms_mask], bins=bins_pms, color="blue", histtype="step",
                   label="Non-PMS", alpha=0.9, linestyle='--')
        ax[0].hist(p_pms[no2m_mask],  bins=bins_pms, color="green", histtype="step",
                   label="No 2MASS Info", alpha=0.9, linestyle=':')
        ax[0].set_xlabel("PMS Probability", fontsize=16)
    
        # (2) log(Age)
        ax[1].hist(p_age[finite_age], bins=bins_age, color="gray", histtype="step",
                   label="All Data", alpha=0.85)
        ax[1].hist(p_age[pms_mask & finite_age],   bins=bins_age, color="orange", histtype="step",
                   label="PMS", alpha=0.9, linestyle='-')
        ax[1].hist(p_age[nopms_mask & finite_age], bins=bins_age, color="blue", histtype="step",
                   label="Non-PMS", alpha=0.9, linestyle='--')
        ax[1].hist(p_age[no2m_mask & finite_age],  bins=bins_age, color="green", histtype="step",
                   label="No 2MASS Info", alpha=0.9, linestyle=':')
        ax[1].set_xlabel("log(Age)", fontsize=16)
        if age_xlim is not None:
            ax[1].set_xlim(age_xlim)
    
        # (3) A_V
        ax[2].hist(p_av[finite_av], bins=bins_av, color="gray", histtype="step",
                   label="All Data", alpha=0.85)
        ax[2].hist(p_av[pms_mask & finite_av],   bins=bins_av, color="orange", histtype="step",
                   label="PMS", alpha=0.9, linestyle='-')
        ax[2].hist(p_av[nopms_mask & finite_av], bins=bins_av, color="blue", histtype="step",
                   label="Non-PMS", alpha=0.9, linestyle='--')
        ax[2].hist(p_av[no2m_mask & finite_av],  bins=bins_av, color="green", histtype="step",
                   label="No 2MASS Info", alpha=0.9, linestyle=':')
        ax[2].set_xlabel(r"$A_V$", fontsize=16)
        if av_xlim is not None:
            ax[2].set_xlim(av_xlim)
    
        # Estética compartida
        for a in ax:
            a.set_ylabel("Count", fontsize=16)
            a.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            a.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            a.tick_params(axis='both', which='both', direction='in', labelsize=14)
            a.grid(False)
    
        ax[1].legend(fontsize=12, framealpha=0.4)
    
        fig.align_labels()
    
        # Guardar si se pide
        if save:
            out = filename or f"pms_stats_cluster{cid}.pdf"
            fig.savefig(os.path.join(self.output_dir, out), dpi="figure", bbox_inches="tight")
    
        plt.show()