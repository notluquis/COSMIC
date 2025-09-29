import logging
from astropy.table import QTable
import numpy as np
import astropy.units as u

class DataLoader:
    """
    A class for loading and managing astronomical data for COSMIC analysis.
    """
    
    PHOTOMETRIC_SYSTEMS = {
        "Gaia": [
            "source_id", "ra", "dec", "l", "b",
            "pm", "pmra", "pmdec", "parallax",
            "pmra_error", "pmdec_error", "parallax_error",
            "Gmag", "G_BPmag", "G_RPmag", 
            "e_Gmag", "e_G_BPmag", "e_G_RPmag",
            "e_BP_RP" 
        ],
        "TMASS": [
            "tmass_designation",
            "j_m", "h_m", "ks_m",
            "j_msigcom", "h_msigcom", "ks_msigcom"
        ],
        "WISE": [
            "allwise_oid",
            "w1mpro", "w2mpro", "w3mpro", "w4mpro",
            "w1mpro_error", "w2mpro_error", "w3mpro_error", "w4mpro_error"
        ],
    }
    GAIA_DISTANCE_COLUMNS = {
        "geometric": ["r_med_geo", "r_lo_geo", "r_hi_geo"],
        "photogeometric": ["r_med_photogeo", "r_lo_photogeo", "r_hi_photogeo"]
    }
    ALIASES = {
        "Gmag":     {"Gmag", "phot_g_mean_mag"},
        "G_BPmag":  {"G_BPmag", "phot_bp_mean_mag"},
        "G_RPmag":  {"G_RPmag", "phot_rp_mean_mag"},
        "e_Gmag":     {"e_Gmag", "phot_g_mean_mag_error", "ephot_g_mean_mag"},
        "e_G_BPmag":  {"e_G_BPmag", "phot_bp_mean_mag_error", "ephot_bp_mean_mag"},
        "e_G_RPmag":  {"e_G_RPmag", "phot_rp_mean_mag_error", "ephot_rp_mean_mag"},
        "e_BP_RP":    {"e_BP_RP", "phot_bp_rp_excess_error"},
        "parallax":        {"parallax"},
        "parallax_error":  {"parallax_error"},
        "pm":              {"pm"},
        "pmra":            {"pmra"},
        "pmra_error":      {"pmra_error"},
        "pmdec":           {"pmdec"},
        "pmdec_error":     {"pmdec_error"},
        "source_id": {"source_id"},
        "ra":        {"ra"},
        "dec":       {"dec"},
        "l":         {"l"},
        "b":         {"b"},
        "tmass_designation": {"tmass_designation"},
        "j_m": {"j_m"},
        "h_m": {"h_m"},
        "ks_m": {"ks_m"},
        "j_msigcom": {"j_msigcom"},
        "h_msigcom": {"h_msigcom"},
        "ks_msigcom": {"ks_msigcom"},
        "nu_eff_used_in_astrometry": {"nu_eff_used_in_astrometry"},
        "pseudocolour": {"pseudocolour"},
        "ecl_lat": {"ecl_lat"},
        "astrometric_params_solved": {"astrometric_params_solved"},
        "r_med_geo": {"r_med_geo"},
        "r_lo_geo": {"r_lo_geo"},
        "r_hi_geo": {"r_hi_geo"},
        "r_med_photogeo": {"r_med_photogeo"},
        "r_lo_photogeo": {"r_lo_photogeo"},
        "r_hi_photogeo": {"r_hi_photogeo"},
        "cluster": {"cluster"},
        "probability_hdbscan": {"probability_hdbscan"},
        "av_sagitta": {"av_sagitta"},
        "pms_sagitta": {"pms_sagitta"},
        "age": {"age"},
        "phot_g_mean_flux": {"phot_g_mean_flux"},
        "phot_g_mean_flux_error": {"phot_g_mean_flux_error"},
        "phot_bp_mean_flux": {"phot_bp_mean_flux"},
        "phot_bp_mean_flux_error": {"phot_bp_mean_flux_error"},
        "phot_rp_mean_flux": {"phot_rp_mean_flux"},
        "phot_rp_mean_flux_error": {"phot_rp_mean_flux_error"},
    }

    def __init__(self, file_path: str, verbose: int = logging.INFO, debug_mode: bool = False):
        self.file_path = file_path
        self.data = None

        self.logger = logging.getLogger("DataLoader")
        self.logger.setLevel(verbose if not debug_mode else logging.DEBUG)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info(f"DataLoader initialized with file path: {self.file_path}")

        if debug_mode:
            self.logger.debug("Debugging mode enabled for DataLoader.")
            astropy_logger = logging.getLogger("astropy")
            astropy_logger.setLevel(logging.DEBUG)
            if not astropy_logger.handlers:
                astropy_handler = logging.StreamHandler()
                astropy_handler.setFormatter(logging.Formatter('%(message)s'))
                astropy_logger.addHandler(astropy_handler)
                
    def _resolve_alias(self, available_cols, canonical_name):
        candidates = self.ALIASES.get(canonical_name, {canonical_name})
        for c in candidates:
            if c in available_cols:
                return c
        return None

    def load_data(self, systems=None, include_distances=None, include_zp_cols=False,
                  include_flux_errors=False, fidelity=None, file_format='ascii.ecsv',
                  probability=None, normalize_names: bool = True) -> QTable:
        if include_distances and not isinstance(include_distances, list):
            raise TypeError("'include_distances' must be a list or None.")

        try:
            full = QTable.read(self.file_path, format=file_format)
            self.logger.info(f"File read successfully: {self.file_path}")
            available = set(full.colnames)
        except Exception as e:
            self.logger.error(f"Failed to read file: {e}")
            raise
        if not systems and not include_distances and not include_zp_cols and not include_flux_errors and not fidelity and not probability:
            self.data = full
            self.logger.info(f"Selected columns loaded: {', '.join(self.data.colnames)}")
            self._handle_masked_data()
            self.conv_wrong_units()
            return self.data
        
        requested = set()
        if systems:
            for sys in systems:
                if sys not in self.PHOTOMETRIC_SYSTEMS:
                    raise ValueError(f"Unknown photometric system: {sys}")
                requested.update(self.PHOTOMETRIC_SYSTEMS[sys])

        if include_distances:
            for d in include_distances:
                if d not in self.GAIA_DISTANCE_COLUMNS:
                    raise ValueError(f"Unknown distance type: {d}")
                requested.update(self.GAIA_DISTANCE_COLUMNS[d])

        if include_zp_cols:
            requested.update(["nu_eff_used_in_astrometry", "pseudocolour", "ecl_lat", "astrometric_params_solved"])

        if include_flux_errors:
            requested.update([
                "phot_g_mean_flux", "phot_g_mean_flux_error",
                "phot_bp_mean_flux", "phot_bp_mean_flux_error",
                "phot_rp_mean_flux", "phot_rp_mean_flux_error"
            ])

        if fidelity and fidelity in available:
            requested.add(fidelity)
            
        if probability and probability in available:
            requested.add(probability)
            
        present_map = {}
        missing = []
        for canon in requested:
            src = self._resolve_alias(available, canon) if normalize_names else (canon if canon in available else None)
            if src is not None:
                present_map[canon] = src
            else:
                missing.append(canon)

        if missing:
            self.logger.warning(
                "The following requested canonical columns were not found (any alias) and will be skipped: "
                + ", ".join(missing)
            )

        cols_present_names = list(set(present_map.values()))
        data = full[cols_present_names].copy()

        self.data = data
        self.logger.info(f"Selected columns loaded: {', '.join(self.data.colnames)}")

        self._handle_masked_data()
        self.conv_wrong_units()

        return self.data

    def _handle_masked_data(self):
        for column_name in self.data.colnames:
            if hasattr(self.data[column_name], "mask"):
                if np.issubdtype(self.data[column_name].dtype, np.integer):
                    self.data[column_name] = self.data[column_name].astype(float)
                self.data[column_name] = self.data[column_name].filled(np.nan)
        self.logger.info("Masked data handled and replaced with NaN.")

    def conv_wrong_units(self):
        unit_corrections = {
            "'electron'.s**-1": u.electron / u.s,
            "log(cm.s**-2)": u.dex(u.cm / u.s**2),
            "'dex'": u.dex,
        }

        for col_name in self.data.colnames:
            if hasattr(self.data[col_name], "unit"):
                current_unit = str(self.data[col_name].unit)
                if current_unit in unit_corrections:
                    corrected_unit = unit_corrections[current_unit]
                    self.data[col_name] = self.data[col_name].value * corrected_unit
                    self.logger.info(f"Updated units for column '{col_name}' to {corrected_unit}.")

    def count_valid_sources(self) -> dict:
        if self.data is None:
            raise ValueError("Data has not been loaded. Use load_data() first.")

        counts = {}

        counts["Gaia IDs"] = (
            np.sum(~np.isnan(self.data["source_id"])) if "source_id" in self.data.colnames else 0
        )

        gaia_photometry_columns = ["phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"]
        counts["Gaia Photometry"] = (
            np.sum(
                np.any(
                    [~np.isnan(self.data[col]) for col in gaia_photometry_columns if col in self.data.colnames],
                    axis=0
                )
            )
            if any(col in self.data.colnames for col in gaia_photometry_columns) else 0
        )

        counts["Gaia Parallaxes"] = (
            np.sum(~np.isnan(self.data["parallax"])) if "parallax" in self.data.colnames else 0
        )

        proper_motion_columns = ["pm", "pmra", "pmdec"]
        counts["Gaia Proper Motions"] = (
            np.sum(
                np.any(
                    [~np.isnan(self.data[col]) for col in proper_motion_columns if col in self.data.colnames],
                    axis=0
                )
            )
            if any(col in self.data.colnames for col in proper_motion_columns) else 0
        )

        counts["TMASS IDs"] = (
            np.sum(
                (self.data["tmass_designation"] != "nan") & (self.data["tmass_designation"] != "")
            ) if "tmass_designation" in self.data.colnames else 0
        )

        tmass_photometry_columns = ["j_m", "h_m", "ks_m"]
        counts["TMASS Photometry"] = (
            np.sum(
                np.any(
                    [~np.isnan(self.data[col]) for col in tmass_photometry_columns if col in self.data.colnames],
                    axis=0
                )
            )
            if any(col in self.data.colnames for col in tmass_photometry_columns) else 0
        )

        counts["WISE IDs"] = (
            np.sum(~np.isnan(self.data["allwise_oid"])) if "allwise_oid" in self.data.colnames else 0
        )

        wise_photometry_columns = ["w1mpro", "w2mpro", "w3mpro", "w4mpro"]
        counts["WISE Photometry"] = (
            np.sum(
                np.any(
                    [~np.isnan(self.data[col]) for col in wise_photometry_columns if col in self.data.colnames],
                    axis=0
                )
            )
            if any(col in self.data.colnames for col in wise_photometry_columns) else 0
        )

        self.logger.info("Valid source counts:")
        for category, count in counts.items():
            self.logger.info(f"- {category}: {count}")

        return counts

    def check_available_photometric_systems(self, file_format: str = 'ascii.ecsv') -> dict:
        try:
            metadata_table = QTable.read(self.file_path, format=file_format)
            column_names = metadata_table.colnames
            self.logger.info(f"File successfully read for photometric system check: {self.file_path}")
        except Exception as e:
            self.logger.error(f"Error reading file header: {e}")
            raise ValueError(f"Error reading the file header: {e}")

        available_systems = {}
        for system, columns in self.PHOTOMETRIC_SYSTEMS.items():
            available_columns = [col for col in columns if col in column_names]
            available_systems[system] = {
                "available": len(available_columns) == len(columns),
                "columns": available_columns,
            }

        self.logger.info("Photometric systems availability check completed.")
        return available_systems
