"""
T24 Model Implementation for Earth MMS Data
=============================================
Based on: Toy-Edens et al. (2024b) - "Classifying 8 years of MMS dayside
plasma regions via unsupervised machine learning"

This script replicates the T24 pipeline for Earth's magnetosphere using
MMS FPI (ions) + FGM (magnetic field) data.

Prerequisites:
    pip install numpy pandas scipy scikit-learn matplotlib pyspedas

Data source:
    MMS data is available via pyspedas (NASA SPEDAS Python interface)
    or from the MMS Science Data Center: https://lasp.colorado.edu/mms/sdc/
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD MMS DATA
# ─────────────────────────────────────────────────────────────

def load_mms_data_pyspedas(trange):
    """
    Download and load MMS1 FPI + FGM data using pyspedas.

    Args:
        trange: list of two strings, e.g. ['2020-01-18', '2020-01-19']

    Returns:
        df: DataFrame with columns: Epoch, Bx, By, Bz, Btot,
            ion_energy_flux (2D: time x energy), ion_energies
    """
    import pyspedas
    from pytplot import get_data

    # Load FGM (magnetometer) at survey mode
    pyspedas.mms.fgm(trange=trange, probe='1', data_rate='srvy', level='l2')
    b_data = get_data('mms1_fgm_b_gsm_srvy_l2')  # (times, [Bx,By,Bz,Btot])

    # Load FPI fast plasma investigation (ions)
    pyspedas.mms.fpi(trange=trange, probe='1', data_rate='fast',
                     level='l2', datatype='dis-moms')
    pyspedas.mms.fpi(trange=trange, probe='1', data_rate='fast',
                     level='l2', datatype='dis-dist')

    ion_eflux = get_data('mms1_dis_energyspectr_omni_fast')  # omni-dir energy flux
    ion_density = get_data('mms1_dis_numberdensity_fast')
    ion_temp = get_data('mms1_dis_temptensor_gse_fast')
    ion_velocity = get_data('mms1_dis_bulkv_gse_fast')

    return b_data, ion_eflux, ion_density, ion_temp, ion_velocity


def resample_to_1min(times, values, resolution_sec=60):
    """
    Resample any time series to 1-minute resolution by taking the mean.
    """
    df = pd.DataFrame({'time': pd.to_datetime(times, unit='s'), 'val': values})
    df = df.set_index('time').resample(f'{resolution_sec}s').mean()
    return df


# ─────────────────────────────────────────────────────────────
# STEP 2: FEATURE ENGINEERING (T24 definitions for MMS)
# ─────────────────────────────────────────────────────────────

def compute_norm_Bt(Btot_series, norm_value=50.0):
    """
    norm_Bt: total B-field normalized to 50 nT (Earth magnetosphere).
    Distinguishes high-B magnetosphere from low-B solar wind / IMF.

    For Mercury (MESSENGER) the paper uses 150 nT instead.
    """
    return Btot_series / norm_value


def compute_ratio_max_width(energy_flux_2d, energies, min_peak_intensity=1.0):
    """
    ratio_max_width: width of the most prominent peak in the ion energy
    spectrum, normalized by the total number of energy bins.

    - Narrow peak (~1 keV) → solar wind
    - Wide/broad distribution → magnetosheath or magnetosphere

    Args:
        energy_flux_2d: np.ndarray of shape (n_times, n_energies)
                        log10 of differential energy flux
        energies:       array of energy bin centers (eV)
        min_peak_intensity: minimum prominence threshold for peak finding

    Returns:
        ratio_max_width: 1D array of length n_times
    """
    n_times, n_energies = energy_flux_2d.shape
    ratio = np.zeros(n_times)

    for i in range(n_times):
        spectrum = energy_flux_2d[i]
        if np.all(np.isnan(spectrum)):
            ratio[i] = np.nan
            continue

        # Find peaks in the log-flux spectrum
        peaks, properties = find_peaks(
            spectrum,
            prominence=min_peak_intensity,
            width=1
        )

        if len(peaks) == 0:
            ratio[i] = 0.0
            continue

        # Select the most prominent peak
        prominences = properties['prominences']
        widths = properties['widths']
        best_idx = np.argmax(prominences)
        peak_width = widths[best_idx]

        ratio[i] = peak_width / n_energies

    return ratio


def compute_ratio_high_low(energy_flux_2d, energies,
                            high_threshold_eV=4000,
                            low_threshold_eV=100):
    """
    ratio_high_low: mean flux at energies >4 keV divided by
    mean flux at energies <100 eV.

    - High value → ion foreshock signature (energetic particles)
    - Low value → solar wind or other regions

    Args:
        energy_flux_2d: np.ndarray (n_times, n_energies), log10 flux
        energies:       1D array of energy bin centers in eV

    Returns:
        ratio_high_low: 1D array of length n_times
    """
    idx_high = energies > high_threshold_eV
    idx_low = energies < low_threshold_eV

    ratio = np.zeros(len(energy_flux_2d))

    for i, spectrum in enumerate(energy_flux_2d):
        if np.all(np.isnan(spectrum)):
            ratio[i] = np.nan
            continue
        mean_high = np.nanmean(spectrum[idx_high]) if idx_high.any() else 0
        mean_low = np.nanmean(spectrum[idx_low]) if idx_low.any() else 0
        ratio[i] = mean_high / mean_low if mean_low > 0 else 0

    return ratio


def compute_spectra_counts(energy_flux_2d):
    """
    spectra_counts: fraction of non-zero (non-NaN) energy bins.
    Used to flag time periods with insufficient data.
    Epochs with spectra_counts < 0.2 are labelled 'unknown'.
    """
    counts = np.sum(~np.isnan(energy_flux_2d) & (energy_flux_2d > 0), axis=1)
    return counts / energy_flux_2d.shape[1]


def apply_magnetosphere_pseudo_feature(norm_Bt, ratio_max_width,
                                       ratio_high_low, peaks_found):
    """
    Pseudo-feature: when no ion spectral peak is detected (peaks_found=False),
    set all three features to 0. This places the data point in a distinct
    location in feature space (the 'magnetosphere corner').
    """
    mask = ~peaks_found
    norm_Bt = norm_Bt.copy()
    ratio_max_width = ratio_max_width.copy()
    ratio_high_low = ratio_high_low.copy()
    norm_Bt[mask] = 0.0
    ratio_max_width[mask] = 0.0
    ratio_high_low[mask] = 0.0
    return norm_Bt, ratio_max_width, ratio_high_low


def build_feature_dataframe(epoch, Btot, energy_flux_2d, energies):
    """
    Full feature engineering pipeline.
    Returns a DataFrame ready for GMM clustering.
    """
    norm_Bt = compute_norm_Bt(Btot)
    ratio_mw = compute_ratio_max_width(energy_flux_2d, energies)
    ratio_hl = compute_ratio_high_low(energy_flux_2d, energies)
    spec_counts = compute_spectra_counts(energy_flux_2d)

    # Determine whether a peak was found at all
    peaks_found = (ratio_mw > 0)

    # Apply pseudo-feature
    norm_Bt, ratio_mw, ratio_hl = apply_magnetosphere_pseudo_feature(
        norm_Bt, ratio_mw, ratio_hl, peaks_found
    )

    df = pd.DataFrame({
        'Epoch': epoch,
        'norm_Btot': norm_Bt,
        'ratio_max_width': ratio_mw,
        'ratio_high_low': ratio_hl,
        'spectra_counts': spec_counts,
        'peaks_found': peaks_found
    })
    return df


# ─────────────────────────────────────────────────────────────
# STEP 3: GAUSSIAN MIXTURE MODEL CLUSTERING
# ─────────────────────────────────────────────────────────────

def train_gmm(feature_df, n_components=4, random_state=42):
    """
    Fit a Gaussian Mixture Model to the three core features.

    For Earth MMS, T24 found 4 clusters:
        0 → solar wind
        1 → ion foreshock
        2 → magnetosheath
        3 → magnetosphere
    (actual mapping depends on your data; verify via inspection)

    Returns:
        gmm: fitted GaussianMixture object
        scaler: fitted StandardScaler
        labels: raw cluster labels (integers)
    """
    features = ['norm_Btot', 'ratio_max_width', 'ratio_high_low']
    X = feature_df[features].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        random_state=random_state,
        n_init=10,
        max_iter=200
    )
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)

    return gmm, scaler, labels


def map_clusters_to_regions(labels, gmm, scaler):
    """
    After fitting, map integer cluster labels to plasma region names.
    Heuristic: inspect cluster means in feature space.

    Cluster characteristics (T24 conventions for Earth):
      - Solar wind:     high ratio_max_width (narrow peak ~1 keV), low norm_Bt
      - Ion foreshock:  high ratio_high_low (energetic tails), low norm_Bt
      - Magnetosheath:  moderate ratio_max_width (broad peak), moderate norm_Bt
      - Magnetosphere:  low ratio_max_width (no clear peak OR pseudo-feature=0),
                        high norm_Bt

    Adjust mapping indices by inspecting cluster centres below.
    """
    # Un-scale cluster means for interpretability
    means_scaled = gmm.means_
    means = scaler.inverse_transform(means_scaled)
    means_df = pd.DataFrame(means,
                             columns=['norm_Btot',
                                      'ratio_max_width',
                                      'ratio_high_low'])
    print("Cluster means (un-scaled):")
    print(means_df.round(4))

    # Auto-heuristic mapping (adjust if needed after visual inspection)
    norm_bt_col = means_df['norm_Btot'].values
    rmw_col = means_df['ratio_max_width'].values
    rhl_col = means_df['ratio_high_low'].values

    # Magnetosphere: highest norm_Bt
    msp_idx = np.argmax(norm_bt_col)
    # Solar wind: narrowest peak (but has a peak) → high ratio_max_width is
    # counterintuitive — in the paper, 'narrow' means small width fraction
    # because the solar wind beam spans only a few energy bins.
    sw_idx = np.argmin(rmw_col[np.arange(len(rmw_col)) != msp_idx])
    # Ion foreshock: highest ratio_high_low
    fs_idx = np.argmax(rhl_col)
    # Magnetosheath: the remaining cluster
    all_idx = set(range(len(means_df)))
    msh_idx = list(all_idx - {msp_idx, sw_idx, fs_idx})[0]

    cluster_map = {
        msp_idx: 'magnetosphere',
        msh_idx: 'magnetosheath',
        sw_idx:  'solar wind',
        fs_idx:  'ion foreshock'
    }
    print("\nCluster → region mapping:", cluster_map)

    named_labels = np.array([cluster_map.get(l, 'unknown') for l in labels])
    return named_labels, cluster_map


# ─────────────────────────────────────────────────────────────
# STEP 4: POST-CLEANING (Earth MMS rules from T24)
# ─────────────────────────────────────────────────────────────

def post_clean_earth(df):
    """
    Post-cleaning rules from T24 for Earth MMS data.
    Modifies 'named_label' in-place.

    Rules (adapt thresholds from T24 supplementary or the original paper):
      1. spectra_counts < 0.2  → unknown
      2. Any non-unknown AND Btot >= 50 nT  → magnetosphere  (strong field)
      3. Magnetosheath AND Btot <= 5 nT     → solar wind     (very weak field)
      4. Solar wind AND ratio_max_width > 0.25 → magnetosheath (broad peak)
      5. Magnetosheath AND ratio_max_width <= 0.25 AND Btot <= 15 nT → solar wind
      6. Remove spurious isolated points (3-min window majority vote)
      7. Remove unphysical transitions (magnetosphere ↔ solar wind)

    Note: exact thresholds for Earth MMS differ from the Mercury values
    in Table 1 of the paper. These are approximate Earth equivalents.
    """
    df = df.copy()

    # Rule 1: insufficient data
    df.loc[df['spectra_counts'] < 0.2, 'named_label'] = 'unknown'

    # Rule 2: very high B → definitively magnetosphere
    mask = (df['named_label'] != 'unknown') & (df['norm_Btot'] * 50 >= 50)
    df.loc[mask, 'named_label'] = 'magnetosphere'

    # Rule 3: magnetosheath with very weak B → solar wind
    mask = (df['named_label'] == 'magnetosheath') & (df['norm_Btot'] * 50 <= 5)
    df.loc[mask, 'named_label'] = 'solar wind'

    # Rule 4: solar wind with wide peak → magnetosheath
    mask = (df['named_label'] == 'solar wind') & (df['ratio_max_width'] > 0.25)
    df.loc[mask, 'named_label'] = 'magnetosheath'

    # Rule 5: magnetosheath with narrow peak AND weak B → solar wind
    mask = ((df['named_label'] == 'magnetosheath') &
            (df['ratio_max_width'] <= 0.25) &
            (df['norm_Btot'] * 50 <= 15))
    df.loc[mask, 'named_label'] = 'solar wind'

    # Rule 6: spurious point removal (3-min window)
    labels = df['named_label'].values
    for i in range(1, len(labels) - 1):
        if labels[i-1] == labels[i+1] and labels[i] != labels[i-1]:
            labels[i] = labels[i-1]
    df['named_label'] = labels

    # Rule 7: remove unphysical magnetosphere ↔ solar wind transitions
    labels = df['named_label'].values
    for i in range(1, len(labels) - 1):
        if ((labels[i-1] == 'magnetosphere' and labels[i] == 'solar wind') or
                (labels[i-1] == 'solar wind' and labels[i] == 'magnetosphere')):
            window = labels[max(0, i-7):min(len(labels), i+8)]
            most_common = pd.Series(window).value_counts().idxmax()
            labels[i] = most_common
    df['named_label'] = labels

    return df


# ─────────────────────────────────────────────────────────────
# STEP 5: IDENTIFY TRANSITIONS (bow shock & magnetopause)
# ─────────────────────────────────────────────────────────────

def identify_transitions(df, min_stable_minutes=10, max_unknown_fraction=0.2):
    """
    Identify magnetopause (Msp ↔ Msh) and bow shock (Msh ↔ SW) crossings.

    Clean transitions require:
      - ≥10 consecutive minutes in region BEFORE the crossing
      - ≥10 consecutive minutes in region AFTER the crossing
      - <20% unknown labels in either window

    Returns:
        transitions: DataFrame with columns [time, type, raw_label]
    """
    labels = df['named_label'].values
    epochs = df['Epoch'].values
    transitions = []

    for i in range(1, len(labels)):
        prev = labels[i-1]
        curr = labels[i]

        # Raw transition
        if prev != curr and prev != 'unknown' and curr != 'unknown':
            if   {prev, curr} == {'magnetosheath', 'solar wind'}:
                t_type = 'bow shock'
            elif {prev, curr} == {'magnetosphere', 'magnetosheath'}:
                t_type = 'magnetopause'
            else:
                t_type = 'unphysical'

            transitions.append({
                'time': epochs[i],
                'type': t_type,
                'from_region': prev,
                'to_region': curr
            })

    trans_df = pd.DataFrame(transitions)

    # Apply stability filter for "clean" transitions
    if len(trans_df) > 0:
        clean_mask = []
        for idx, row in trans_df.iterrows():
            i = np.searchsorted(epochs, row['time'])
            # Check pre-window
            pre_window = labels[max(0, i-min_stable_minutes):i]
            post_window = labels[i:min(len(labels), i+min_stable_minutes)]
            pre_unknown = np.sum(pre_window == 'unknown') / len(pre_window) if len(pre_window) > 0 else 1
            post_unknown = np.sum(post_window == 'unknown') / len(post_window) if len(post_window) > 0 else 1
            pre_stable = np.all(pre_window[pre_window != 'unknown'] == row['from_region'])
            post_stable = np.all(post_window[post_window != 'unknown'] == row['to_region'])
            is_clean = (pre_stable and post_stable and
                        pre_unknown < max_unknown_fraction and
                        post_unknown < max_unknown_fraction)
            clean_mask.append(is_clean)
        trans_df['is_clean'] = clean_mask

    return trans_df


# ─────────────────────────────────────────────────────────────
# STEP 6: FULL PIPELINE RUNNER
# ─────────────────────────────────────────────────────────────

def run_t24_pipeline(epoch, Btot, energy_flux_2d, energies,
                     n_gmm_components=4, dayside_only=True, X_mso=None):
    """
    Run the complete T24 pipeline from raw MMS data to labelled regions.

    Args:
        epoch:           1D array of timestamps (datetime64 or Unix seconds)
        Btot:            1D array of total B-field magnitude (nT), 1-min res
        energy_flux_2d:  2D array (n_times x n_energies), log10 ion energy flux
        energies:        1D array of energy bin centers (eV)
        n_gmm_components: number of GMM clusters (default 4 for Earth)
        dayside_only:    if True, filter to X_mso > 0 (dayside)
        X_mso:           optional X coordinate in MSO/GSE (for dayside filter)

    Returns:
        result_df: DataFrame with all features, raw labels, cleaned labels,
                   and transitions
    """
    print("=== T24 Earth MMS Pipeline ===\n")

    # Dayside filter
    if dayside_only and X_mso is not None:
        mask = X_mso >= 0
        epoch = epoch[mask]
        Btot = Btot[mask]
        energy_flux_2d = energy_flux_2d[mask]
        print(f"Dayside filter: kept {mask.sum()} / {len(mask)} epochs")

    # Feature engineering
    print("Step 1/4: Engineering features...")
    feat_df = build_feature_dataframe(epoch, Btot, energy_flux_2d, energies)

    # GMM clustering
    print("Step 2/4: Fitting GMM...")
    valid = feat_df.dropna(subset=['norm_Btot', 'ratio_max_width', 'ratio_high_low'])
    gmm, scaler, raw_labels = train_gmm(valid)
    named_labels, cluster_map = map_clusters_to_regions(raw_labels, gmm, scaler)
    feat_df.loc[valid.index, 'raw_named_label'] = named_labels

    # Fill missing as unknown
    feat_df['raw_named_label'] = feat_df['raw_named_label'].fillna('unknown')
    feat_df['named_label'] = feat_df['raw_named_label'].copy()

    # Post-cleaning
    print("Step 3/4: Applying post-cleaning rules...")
    feat_df = post_clean_earth(feat_df)

    # Identify transitions
    print("Step 4/4: Identifying transitions...")
    trans_df = identify_transitions(feat_df)
    bs = trans_df[trans_df['type'] == 'bow shock']
    mp = trans_df[trans_df['type'] == 'magnetopause']
    clean_bs = bs[bs['is_clean']] if 'is_clean' in bs.columns else bs
    clean_mp = mp[mp['is_clean']] if 'is_clean' in mp.columns else mp
    print(f"  Bow shock crossings:   {len(bs)} raw, {len(clean_bs)} clean")
    print(f"  Magnetopause crossings: {len(mp)} raw, {len(clean_mp)} clean")

    print("\nRegion counts:")
    print(feat_df['named_label'].value_counts())

    return feat_df, trans_df


# ─────────────────────────────────────────────────────────────
# STEP 7: LOAD REAL MMS DATA FROM CSV
# ─────────────────────────────────────────────────────────────

def load_labeled_sunside_csv(csv_path='labeled_sunside_data.csv', limit=None, dayside_only=True):
    """
    Load pre-processed MMS dayside data from CSV.
    
    Args:
        csv_path: path to labeled_sunside_data.csv
        limit: max samples (None = all)
        dayside_only: filter to x_mso > 0
        
    Returns:
        df: DataFrame with features ready for analysis
    """
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path, nrows=limit)
    
    if dayside_only:
        df = df[df['x_mso'] > 0].reset_index(drop=True)
        print(f"  Dayside filter: {len(df)} samples")
    
    df['Epoch'] = pd.to_datetime(df['Epoch'], format='%d-%m-%Y %H:%M')
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['Epoch'].min()} to {df['Epoch'].max()}")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 8: QUICK-START WITH SYNTHETIC DATA (for testing)
# ─────────────────────────────────────────────────────────────

def generate_synthetic_mms_data(n_hours=6, seed=42):
    """
    Generate synthetic MMS-like data to test the pipeline without
    downloading real data. Mimics a typical dayside orbit:
    SW → Msh → Msp → Msh → SW
    """
    rng = np.random.default_rng(seed)
    n_min = n_hours * 60
    t = np.arange(n_min)
    n_energies = 64

    # Regions: SW (0-60min), Msh (60-120), Msp (120-240), Msh (240-300), SW (300+)
    def region_at(t_i):
        if t_i < 60 or t_i >= 300:
            return 'solar wind'
        elif t_i < 120 or t_i >= 240:
            return 'magnetosheath'
        else:
            return 'magnetosphere'

    energies = np.logspace(1, 4.1, n_energies)  # 10 eV to ~13 keV

    Btot = np.zeros(n_min)
    energy_flux = np.full((n_min, n_energies), np.nan)

    for i in range(n_min):
        r = region_at(i)
        noise = rng.normal

        if r == 'solar wind':
            # Narrow beam ~1 keV, low B
            Btot[i] = rng.normal(5, 1)
            peak_bin = np.argmin(np.abs(energies - 1000))
            spectrum = np.zeros(n_energies)
            spectrum[peak_bin-2:peak_bin+3] = rng.normal(6, 0.3, 5)
            energy_flux[i] = np.clip(spectrum, 0, None)

        elif r == 'magnetosheath':
            # Broad heated distribution ~100eV-2keV, moderate B
            Btot[i] = rng.normal(20, 4)
            peak_bin = np.argmin(np.abs(energies - 500))
            spectrum = np.zeros(n_energies)
            spectrum[peak_bin-8:peak_bin+8] = rng.normal(5, 0.4, 16)
            energy_flux[i] = np.clip(spectrum, 0, None)

        else:  # magnetosphere
            # Weak or no clear peak, high B
            Btot[i] = rng.normal(60, 8)
            energy_flux[i] = rng.normal(2, 0.5, n_energies).clip(0)

    Btot = np.abs(Btot)
    epoch = np.array([np.datetime64('2020-01-18') +
                       np.timedelta64(int(m), 'm') for m in t])
    true_labels = [region_at(i) for i in t]

    return epoch, Btot, energy_flux, energies, true_labels


# ─────────────────────────────────────────────────────────────
# MAIN — run with synthetic data by default
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--csv':
        # Run with real data from CSV
        print("="*80)
        print("T24 PIPELINE - REAL MMS DATA FROM CSV")
        print("="*80 + "\n")
        
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
        df_real = load_labeled_sunside_csv('labeled_sunside_data.csv', limit=limit, dayside_only=True)
        
        print("\nExisting classifications (from CSV):")
        print(df_real['named_label'].value_counts())
        
        print("\nRegion distribution:")
        print(df_real.groupby('named_label').agg({
            'Epoch': 'count',
            'norm_Btot': ['min', 'max', 'mean'],
            'ratio_max_width': ['min', 'max', 'mean'],
            'ratio_high_low': ['min', 'max', 'mean']
        }).round(2))
        
        # Save summary
        df_real.to_csv('t24_real_results.csv', index=False)
        print("\n✓ Saved: t24_real_results.csv")
        
    else:
        # Run with synthetic data (default)
        print("Running T24 pipeline on synthetic MMS data...\n")
        epoch, Btot, eflux, energies, true_labels = generate_synthetic_mms_data()

        result_df, trans_df = run_t24_pipeline(
            epoch, Btot, eflux, energies, dayside_only=False
        )

        # Save outputs
        result_df.to_csv('t24_earth_results.csv', index=False)
        trans_df.to_csv('t24_earth_transitions.csv', index=False)
        print("\nSaved: t24_earth_results.csv, t24_earth_transitions.csv")

        # ── Quick accuracy check vs synthetic ground truth ──
        result_df['true_label'] = true_labels
        valid = result_df[result_df['named_label'] != 'unknown']
        accuracy = (valid['named_label'] == valid['true_label']).mean()
        print(f"\nAccuracy vs ground truth: {accuracy:.1%}")

        # ── Plot ──
        fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(result_df['Epoch'], result_df['norm_Btot'], lw=0.8)
        axes[0].set_ylabel('norm_Bt')
        axes[1].plot(result_df['Epoch'], result_df['ratio_max_width'], lw=0.8, color='orange')
        axes[1].set_ylabel('ratio_max_width')
        axes[2].plot(result_df['Epoch'], result_df['ratio_high_low'], lw=0.8, color='green')
        axes[2].set_ylabel('ratio_high_low')

        region_cmap = {'solar wind': 0, 'magnetosheath': 1, 'magnetosphere': 2,
                       'ion foreshock': 3, 'unknown': -1}
        colors = result_df['named_label'].map(region_cmap).fillna(-1)
        axes[3].scatter(result_df['Epoch'], colors, c=colors, s=2, cmap='tab10')
        axes[3].set_yticks([0, 1, 2, 3])
        axes[3].set_yticklabels(['SW', 'Msh', 'Msp', 'Foreshock'])
        axes[3].set_ylabel('Region')
        plt.tight_layout()
        plt.savefig('t24_earth_classification.png', dpi=120)
        print("Saved: t24_earth_classification.png")
