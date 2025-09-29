import numpy as np
import astropy.units as u
from astropy.table import QTable
from zero_point import zpt
import warnings

warnings.filterwarnings(
    "ignore",
    message="'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8.",
    category=FutureWarning,
    module="sklearn.utils.deprecation"
)
class DataPreprocessor:
    def __init__(self, data: QTable):
        self.data = data
        self.total_count = len(data)
        zpt.load_tables()

    def fill_missing_values(self):
        for column_name in self.data.colnames:
            if hasattr(self.data[column_name], 'mask'):
                if np.issubdtype(self.data[column_name].dtype, np.integer):
                    self.data[column_name] = self.data[column_name].astype(float)
                self.data[column_name] = self.data[column_name].filled(np.nan)
        print("Filled missing values in masked columns.")

    def rename_columns(self):
        column_mapping = {
            'phot_g_mean_mag': 'Gmag',
            'phot_bp_mean_mag': 'G_BPmag',
            'phot_rp_mean_mag': 'G_RPmag',
            'bp_rp': 'BP_RP'
        }
        for old_name, new_name in column_mapping.items():
            if old_name in self.data.colnames:
                self.data.rename_column(old_name, new_name)
        print("Renamed columns for standardization.")

    def apply_zero_point_correction(self):
        required_columns = ['parallax', 'Gmag', 'nu_eff_used_in_astrometry', 'pseudocolour', 
                            'ecl_lat', 'astrometric_params_solved']
    
        missing_columns = [col for col in required_columns if col not in self.data.colnames]
        if missing_columns:
            warnings.warn(f"Missing columns for zero-point correction: {missing_columns}. Skipping correction.", UserWarning)
            return
    
        pseudocolour_range = (1.24 / u.um, 1.72 / u.um)
        pseudocolour = self.data['pseudocolour']
    
        out_of_range = (pseudocolour < pseudocolour_range[0]) | (pseudocolour > pseudocolour_range[1])
        if np.any(out_of_range):
            out_of_range_count = np.sum(out_of_range)
            warnings.warn(
                f"{out_of_range_count} source(s) have pseudocolour values outside the expected range "
                f"{pseudocolour_range[0]} - {pseudocolour_range[1]}. Maximum corrections are applied at the range boundaries.",
                UserWarning
            )
    
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.data['parallax_observed'] = self.data['parallax']
            self.data['zpvals'] = zpt.get_zpt(
                self.data['Gmag'], self.data['nu_eff_used_in_astrometry'],
                self.data['pseudocolour'], self.data['ecl_lat'], 
                self.data['astrometric_params_solved']
            ) * u.mas
    
        self.data['zpvals'] = np.ma.masked_invalid(self.data['zpvals']).filled(0)
        self.data['parallax'] = self.data['parallax_observed'] - self.data['zpvals']
        print("Applied zero-point correction to parallax values.")

    def correct_proper_motion(self):
        self.data['pmra_obs'], self.data['pmdec_obs'] = self.data['pmra'], self.data['pmdec']

        table1 = np.array([
            [0.0, 9.0, 18.4, 33.8, -11.3],
            [9.0, 9.5, 14.0, 30.7, -19.4],
            [9.5, 10.0, 12.8, 31.4, -11.8],
            [10.0, 10.5, 13.6, 35.7, -10.5],
            [10.5, 11.0, 16.2, 50.0, 2.1],
            [11.0, 11.5, 19.4, 59.9, 0.2],
            [11.5, 11.75, 21.8, 64.2, 1.0],
            [11.75, 12.0, 17.7, 65.6, -1.9],
            [12.0, 12.25, 21.3, 74.8, 2.1],
            [12.25, 12.5, 25.7, 73.6, 1.0],
            [12.5, 12.75, 27.3, 76.6, 0.5],
            [12.75, 13.0, 34.9, 68.9, -2.9]
        ]).T

        Gmin, Gmax, omegaX, omegaY, omegaZ = table1
        G = self.data['Gmag']
        ra, dec = self.data['ra'], self.data['dec']
        pmra, pmdec = self.data['pmra'], self.data['pmdec']

        pmra_corr = np.zeros(len(G))
        pmdec_corr = np.zeros(len(G))

        mask = G < 13 * u.mag
        for i in np.where(mask)[0]:
            g_val = G[i].value
            idx = np.where((Gmin <= g_val) & (Gmax > g_val))[0][0]
            oX, oY, oZ = omegaX[idx], omegaY[idx], omegaZ[idx]

            ra_val, dec_val = np.radians(ra[i]), np.radians(dec[i])
            pmra_corr[i] = -np.sin(dec_val) * np.cos(ra_val) * oX - np.sin(dec_val) * np.sin(ra_val) * oY + np.cos(dec_val) * oZ
            pmdec_corr[i] = np.sin(ra_val) * oX - np.cos(ra_val) * oY

        self.data['pmra'] = pmra - pmra_corr / 1000.0 * (u.mas / u.yr)
        self.data['pmdec'] = pmdec - pmdec_corr / 1000.0 * (u.mas / u.yr)
        print("Transformed proper motion to ICRF frame.")

    def add_photometric_errors(self):
        def calculate_mag_error(flux, flux_error):
            flux = flux.value * (u.electron / u.s)
            flux_error = flux_error.value * (u.electron / u.s)
            return 2.5 * np.log10(1 + (flux_error / flux)) * u.mag

        self.data['e_Gmag'] = calculate_mag_error(self.data['phot_g_mean_flux'], self.data['phot_g_mean_flux_error'])
        self.data['e_G_BPmag'] = calculate_mag_error(self.data['phot_bp_mean_flux'], self.data['phot_bp_mean_flux_error'])
        self.data['e_G_RPmag'] = calculate_mag_error(self.data['phot_rp_mean_flux'], self.data['phot_rp_mean_flux_error'])
        self.data['e_BP_RP'] = np.sqrt(self.data['e_G_BPmag']**2 + self.data['e_G_RPmag']**2)
        print("Added photometric errors to the data.")
    def drop_invalid_sources(self, columns_to_check=None):
        if columns_to_check is None:
            columns_to_check = ['ra', 'dec', 'pmra', 'pmdec', 'parallax', 'Gmag', 'G_BPmag', 'G_RPmag']
        
        missing_columns = [col for col in columns_to_check if col not in self.data.colnames]
        if missing_columns:
            raise ValueError(f"Columns not found in data: {missing_columns}")
        
        mask = np.ones(len(self.data), dtype=bool)
        
        for column in columns_to_check:
            mask &= np.isfinite(self.data[column])
        
        invalid_count = len(self.data) - np.sum(mask)
        self.data = self.data[mask]
        
        print(f"Dropped {invalid_count} sources with invalid values in {columns_to_check}.")
    def filter_data(self, fidelity_column='fidelity_v2', fidelity_threshold=0.5):
        if fidelity_column not in self.data.colnames:
            raise ValueError(f"Column '{fidelity_column}' not found in the data.")
        
        high_fidelity_mask = self.data[fidelity_column] > fidelity_threshold
        low_fidelity_mask = ~high_fidelity_mask
        
        good_data = self.data[high_fidelity_mask]
        bad_data = self.data[low_fidelity_mask]
        
        def count_tmass_sources(data):
            return np.sum(data['tmass_designation'] != 'nan') if 'tmass_designation' in data.colnames else 0
    
        def count_wise_sources(data):
            return np.sum(~np.isnan(data['allwise_oid'])) if 'allwise_oid' in data.colnames else 0
    
        tmass_good_count = count_tmass_sources(good_data)
        tmass_bad_count = count_tmass_sources(bad_data)
        
        wise_good_count = count_wise_sources(good_data)
        wise_bad_count = count_wise_sources(bad_data)
        
        print(f"Filtered data based on fidelity threshold of {fidelity_threshold}.")
        print(f"High-fidelity sources: {len(good_data)}, Low-fidelity sources: {len(bad_data)}, {round(len(good_data) / len(self.data) * 100)}% of good sources.")
        print(f"High-fidelity sources with 2MASS data: {tmass_good_count}")
        print(f"Low-fidelity sources with 2MASS data: {tmass_bad_count}")
        print(f"High-fidelity sources with WISE data: {wise_good_count}")
        print(f"Low-fidelity sources with WISE data: {wise_bad_count}")
        
        return good_data, bad_data
