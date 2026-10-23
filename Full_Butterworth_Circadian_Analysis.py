import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
from scipy.signal import butter, filtfilt
import os
import gc
import glob

# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    'sampling_rate': 0.5,
    't_min': 0,
    't_max': 72,
    'sep': ';',
    'decimal': ',',
    'min_period': 1,
    'max_period': 40,
    'num_scales': 40,
    'wt_std_multiplier': 2,
    'outlier_std_multiplier': 6,
    'highpass_cutoff_period': 40,
    'highpass_order': 4,
}

# =============================================================================
# FILTERING MODELS (BUTTERWORTH)
# =============================================================================

def filter_biolum_trend(time, intensity, cutoff_period=None, order=None):
    if cutoff_period is None:
        cutoff_period = CONFIG['highpass_cutoff_period']
    if order is None:
        order = CONFIG['highpass_order']
    dt = np.mean(np.diff(time))
    fs = 1 / dt
    nyquist = fs / 2
    normalized_cutoff = min((1 / cutoff_period) / nyquist, 0.99)
    b, a = butter(order, normalized_cutoff, btype='high')
    default_padlen = 3 * max(len(a), len(b))
    if len(intensity) > default_padlen:
        detrended_intensity = filtfilt(b, a, intensity)
    elif len(intensity) > 1:
        detrended_intensity = filtfilt(b, a, intensity, padlen=len(intensity) - 1)
    else:
        detrended_intensity = intensity - np.mean(intensity)
    trend_intensity = intensity - detrended_intensity
    rate_of_change = np.gradient(intensity, time)
    trend_rate = np.gradient(trend_intensity, time)
    return trend_intensity, detrended_intensity, rate_of_change, trend_rate

# =============================================================================
# WAVELET ANALYSIS + FEATURE EXTRACTION
# =============================================================================

def compute_cwt(time, signal):
    dt = np.mean(np.diff(time))
    norm_signal = signal - np.mean(signal)
    scales = np.linspace(CONFIG['min_period'] / dt, CONFIG['max_period'] / dt, CONFIG['num_scales'])
    coeffs, freqs = pywt.cwt(norm_signal, scales, 'morl', sampling_period=dt)
    power = np.abs(coeffs)**2
    periods = 1 / freqs
    gws = power.mean(axis=1)
    return periods, power, gws

def extract_wavelet_features(periods, power, gws):
    peak_idx = np.argmax(gws)
    peak_period = periods[peak_idx]
    peak_power = gws[peak_idx]
    half_max = peak_power / 2
    indices = np.where(gws > half_max)[0]
    bandwidth = periods[indices[-1]] - periods[indices[0]] if len(indices) > 1 else 0
    mean_power = np.mean(gws)
    peak_to_mean_ratio = peak_power / mean_power if mean_power > 0 else 0.0
    ridge_strength = np.mean(power[peak_idx, :])
    max_power = np.max(gws)
    min_power = np.min(gws)
    power_variance = np.var(gws)
    return {
        "peak_period": peak_period,
        "peak_power": peak_power,
        "peak_to_mean_ratio": peak_to_mean_ratio,
        "bandwidth": bandwidth,
        "ridge_strength": ridge_strength,
        "mean_power": mean_power,
        "max_power": max_power,
        "min_power": min_power,
        "power_variance": power_variance
    }

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_4panel(time, raw, trend, rate, trend_rate, detrended, periods, power, gws, peak_period, title, data_type, save_path):
    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(title, fontsize=16, fontweight="bold")
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time, raw, color="blue")
    ax1.plot(time, trend, linestyle="--", color="red")
    ax1.set_title("Raw + Trend")
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time, rate, color="blue")
    ax2.plot(time, trend_rate, linestyle="--", color="red")
    ax2.set_title("Rate of Change")
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(time, detrended, color="purple")
    ax3.axhline(0, color="black", linestyle="--")
    ax3.set_title("Detrended Signal")
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    gs_wave = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 1], width_ratios=[5, 1])
    ax4 = fig.add_subplot(gs_wave[0])
    ax4.pcolormesh(time, periods, power, cmap="viridis", shading="auto")
    ax4.axhline(peak_period, color="white", ls="--")
    ax4.set_title("Wavelet Spectrum")
    ax4.invert_yaxis()
    ax4_gws = fig.add_subplot(gs_wave[1], sharey=ax4)
    ax4_gws.plot(gws, periods, color="black")
    ax4_gws.axhline(peak_period, color="black", ls="--")
    ax4_gws.set_title("GWS")
    ax4_gws.invert_yaxis()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    gc.collect()

def plot_plate_grid(time, well_traces, col_names, title, save_path, ylim=None):
    fig, axes = plt.subplots(8, 12, figsize=(24, 14), sharex=True, sharey=True)
    fig.suptitle(title, fontsize=16, fontweight="bold")
    for i in range(96):
        row, c = divmod(i, 12)
        ax = axes[row, c]
        col = col_names[i]
        traces = well_traces[i]
        if traces and col is not None:
            has_valid = False
            for trace, is_wt, strain_name, run_nr in traces:
                if not np.all(np.isnan(trace)):
                    color = "red" if is_wt else "purple"
                    ax.plot(time, trace, color=color, linewidth=0.8, alpha=0.7)
                    has_valid = True
            if has_valid:
                ax.axhline(0, color="black", linestyle="--", linewidth=0.4, alpha=0.5)
                ax.set_title(col, fontsize=5, pad=1)
            else:
                ax.set_title(f"{col} (Skipped)", fontsize=5, pad=1, color="gray")
        else:
            ax.set_title("Empty", fontsize=5, pad=1, color="gray")
    for i in range(96):
        row, c = divmod(i, 12)
        ax = axes[row, c]
        ax.set_xticks([])
        ax.set_yticks([])
        if c == 0:
            ax.set_ylabel(chr(65 + row), fontsize=9, rotation=0, labelpad=12)
        if row == 0:
            ax.text(0.5, 1.45, str(c + 1), transform=ax.transAxes, ha="center", fontsize=9)
    if ylim is not None:
        axes[0, 0].set_ylim(ylim)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

def plot_wt_plate_grid(time, wt_traces, title, save_path, ylim=None):
    fig, axes = plt.subplots(8, 12, figsize=(24, 14), sharex=True, sharey=True)
    fig.suptitle(title, fontsize=16, fontweight="bold")
    for i in range(min(96, len(wt_traces))):
        row, c = divmod(i, 12)
        ax = axes[row, c]
        traces = wt_traces[i]
        if traces:
            has_valid = False
            for trace, strain_name, run_nr in traces:
                if not np.all(np.isnan(trace)):
                    ax.plot(time, trace, color="red", linewidth=0.9, alpha=0.85)
                    has_valid = True
            if has_valid:
                ax.axhline(0, color="black", linestyle="--", linewidth=0.4, alpha=0.5)
                ax.set_title(f"WT (Run {traces[0][2]})", fontsize=5, pad=1)
            else:
                ax.set_title("WT (Skipped)", fontsize=5, pad=1, color="gray")
        else:
            ax.set_title("Empty", fontsize=5, pad=1, color="gray")
    for i in range(96):
        row, c = divmod(i, 12)
        ax = axes[row, c]
        ax.set_xticks([])
        ax.set_yticks([])
        if c == 0:
            ax.set_ylabel(chr(65 + row), fontsize=9, rotation=0, labelpad=12)
        if row == 0:
            ax.text(0.5, 1.45, str(c + 1), transform=ax.transAxes, ha="center", fontsize=9)
    if ylim is not None:
        axes[0, 0].set_ylim(ylim)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

def plot_traces_with_mean_std(ax, time, matrix, color, title):
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    for trace in matrix:
        ax.plot(time, trace, color=color, alpha=0.25, linewidth=0.8)
    ax.fill_between(time, mean - std, mean + std, color="gray", alpha=0.4)
    ax.plot(time, mean, color=color, linewidth=2)
    ax.set_title(title)

def plot_wt_aggregate(time, raw_matrix, rate_matrix, detrended_matrix, periods_list, gws_list, title, data_type, save_path):
    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(title, fontsize=16, fontweight="bold")
    gs = fig.add_gridspec(2, 2)
    n = raw_matrix.shape[0]
    ax1 = fig.add_subplot(gs[0, 0])
    plot_traces_with_mean_std(ax1, time, raw_matrix, "blue", f"WT Raw (n={n}, mean ± SD)")
    ax2 = fig.add_subplot(gs[0, 1])
    plot_traces_with_mean_std(ax2, time, rate_matrix, "orange", f"WT Rate (n={n}, mean ± SD)")
    ax3 = fig.add_subplot(gs[1, 0])
    plot_traces_with_mean_std(ax3, time, detrended_matrix, "purple", f"WT Detrended (n={n}, mean ± SD)")
    ax3.axhline(0, color="black", linestyle="--", linewidth=0.8)
    periods = periods_list[0]
    gws_matrix = np.vstack(gws_list)
    mean_gws = np.mean(gws_matrix, axis=0)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(mean_gws, periods, color="black")
    ax4.set_title("WT Mean GWS")
    ax4.invert_yaxis()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_clustering_separate(all_gene_results, output_dir, custom_wt_stats=None):
    if not all_gene_results:
        return
    from matplotlib.patches import Rectangle
    df = pd.DataFrame(all_gene_results)
    datasets = [
        "Colony Area Expansion",
        "Mean Colony Bioluminescence Intensity",
        "Mean Edge Bioluminescence Intensity"
    ]
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
    outlier_n_std = CONFIG['outlier_std_multiplier']

    for i, ds in enumerate(datasets):
        sub = df[df["dataset"] == ds].copy()
        if sub.empty:
            continue

        if custom_wt_stats and ds in custom_wt_stats:
            period_mean, period_std, power_mean, power_std = custom_wt_stats[ds]
            n_std = CONFIG['wt_std_multiplier']
        else:
            wt_sub = sub[sub["type"] == "WT"]
            if len(wt_sub) > 1:
                n_std = CONFIG['wt_std_multiplier']
                period_mean, period_std = wt_sub["peak_period"].mean(), wt_sub["peak_period"].std()
                power_mean, power_std = wt_sub["peak_power"].mean(), wt_sub["peak_power"].std()
            else:
                n_std = CONFIG['wt_std_multiplier']
                period_mean, period_std = None, None
                power_mean, power_std = sub["peak_power"].mean(), sub["peak_power"].std()

        if power_std and not np.isnan(power_std) and power_std > 0:
            power_ceiling = power_mean + outlier_n_std * power_std
        else:
            power_ceiling = sub["peak_power"].max()

        sub["is_outlier"] = sub["peak_power"] > power_ceiling
        sub["plot_power"] = np.minimum(sub["peak_power"], power_ceiling)
        wt_sub = sub[sub["type"] == "WT"]
        del_sub = sub[sub["type"] == "Deletion"]

        fig, ax = plt.subplots(figsize=(8, 6))

        if period_mean is not None and period_std is not None and not np.isnan(period_std):
            box = Rectangle(
                (period_mean - n_std * period_std, power_mean - n_std * power_std),
                2 * n_std * period_std, 2 * n_std * power_std,
                facecolor="red", alpha=0.12, edgecolor="red",
                linestyle="--", linewidth=1, label=f"WT mean ± {n_std} SD", zorder=1
            )
            ax.add_patch(box)

        is_arrhythmic = sub["is_arrhythmic"].fillna(False) if "is_arrhythmic" in sub else pd.Series(False, index=sub.index)
        plain = ~sub["is_outlier"] & ~is_arrhythmic
        arrhythmic_normal = ~sub["is_outlier"] & is_arrhythmic

        del_normal = del_sub[plain[del_sub.index]]
        wt_normal = wt_sub[plain[wt_sub.index]]
        arrhythmic_pts = sub[arrhythmic_normal]
        outliers = sub[sub["is_outlier"]]

        ax.scatter(del_normal["peak_period"], del_normal["plot_power"], color=colors[i], alpha=0.7, s=60, label="Deletion", zorder=2)
        ax.scatter(wt_normal["peak_period"], wt_normal["plot_power"], color="red", alpha=0.9, s=60, label="WT", zorder=3)

        if not arrhythmic_pts.empty:
            ax.scatter(arrhythmic_pts["peak_period"], arrhythmic_pts["plot_power"], color="gray", marker="D", s=55, edgecolor="black", linewidth=0.6, alpha=0.85, label="Arrhythmic", zorder=3.5)

        if not outliers.empty:
            ax.scatter(outliers["peak_period"], outliers["plot_power"], color="black", marker="X", s=90, edgecolor="orange", linewidth=1.2, label=f"Outlier (> {outlier_n_std} SD, clipped)", zorder=4)

        ax.axvline(24, color="black", ls="--", alpha=0.5, label="24 h")

        current_bottom, _ = ax.get_ylim()
        margin = 0.5 * power_std if power_std and not np.isnan(power_std) and power_std > 0 else 0.05 * power_ceiling
        ax.set_ylim(bottom=current_bottom, top=power_ceiling + margin)

        ax.set_xlabel("Peak Period (h)")
        ax.set_ylabel("Peak Power (GWS amplitude)")
        ax.set_title(f"Rhythmicity Clustering — {ds}")
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend()

        save_path = os.path.join(output_dir, f"cluster_{ds.replace(' ', '_')}.png")
        os.makedirs(output_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

def plot_period_correlation(all_gene_results, output_dir):
    if not all_gene_results:
        return
    df = pd.DataFrame(all_gene_results)
    idx_cols = ["plate_id", "strain"] if "plate_id" in df.columns else "strain"
    pivot = df.pivot_table(index=idx_cols, columns="dataset", values="peak_period", aggfunc="first")
    type_map = df.drop_duplicates(idx_cols).set_index(idx_cols)["type"]
    pivot["type"] = type_map

    pairs = [
        ("Colony Area Expansion", "Mean Colony Bioluminescence Intensity", "CAE_vs_MCBI", "CAE Peak Period (h)", "MCBI Peak Period (h)"),
        ("Mean Edge Bioluminescence Intensity", "Mean Colony Bioluminescence Intensity", "MEBI_vs_MCBI", "MEBI Peak Period (h)", "MCBI Peak Period (h)"),
        ("Colony Area Expansion", "Mean Edge Bioluminescence Intensity", "CAE_vs_MEBI", "CAE Peak Period (h)", "MEBI Peak Period (h)")
    ]

    for x_ds, y_ds, tag, xlabel, ylabel in pairs:
        if x_ds not in pivot.columns or y_ds not in pivot.columns:
            continue
        sub = pivot[[x_ds, y_ds, "type"]].dropna(subset=[x_ds, y_ds])
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 7))
        wt_sub = sub[sub["type"] == "WT"]
        del_sub = sub[sub["type"] == "Deletion"]
        ax.scatter(del_sub[x_ds], del_sub[y_ds], color="steelblue", alpha=0.6, s=50, label="Deletion", zorder=2)
        ax.scatter(wt_sub[x_ds], wt_sub[y_ds], color="red", alpha=0.9, s=60, label="WT", zorder=3)
        lims = [min(sub[x_ds].min(), sub[y_ds].min()), max(sub[x_ds].max(), sub[y_ds].max())]
        ax.plot(lims, lims, color="black", ls="--", alpha=0.5, label="y = x", zorder=1)
        title = f"Peak Period Correlation\n{tag.replace('_', ' ')}"
        if len(sub) > 1:
            r = sub[x_ds].corr(sub[y_ds])
            title += f" (r = {r:.2f}, n = {len(sub)})"
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend()
        save_path = os.path.join(output_dir, f"correlation_{tag}.png")
        os.makedirs(output_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

def save_results_to_excel(all_results, output_dir, filename="Rhythmicity_Results.xlsx"):
    if not all_results:
        return
    df = pd.DataFrame(all_results)
    cols = df.columns.tolist()
    if 'run_nr' in cols:
        cols.insert(0, cols.pop(cols.index('run_nr')))
    if 'plate_id' in cols:
        cols.insert(0, cols.pop(cols.index('plate_id')))
        df = df[cols]
    excel_path = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for dataset in df["dataset"].unique():
            sub = df[df["dataset"] == dataset]
            sheet_name = dataset[:31]
            sub.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Saved Excel file to: {excel_path}")

def label_phenotypes(all_gene_results, custom_wt_stats=None):
    if not all_gene_results:
        return pd.DataFrame()
    df = pd.DataFrame(all_gene_results).copy()
    n_std = CONFIG['wt_std_multiplier']
    df["phenotype"] = ""
    group_cols = ["dataset", "run_nr"] if "run_nr" in df.columns else ["dataset"]

    for group_keys, sub in df.groupby(group_cols):
        ds = group_keys[0] if isinstance(group_keys, tuple) else group_keys
        wt_stats = None
        if custom_wt_stats:
            if group_keys in custom_wt_stats:
                wt_stats = custom_wt_stats[group_keys]
            elif ds in custom_wt_stats:
                wt_stats = custom_wt_stats[ds]

        if wt_stats:
            period_mean, period_std, power_mean, power_std = wt_stats
            period_low, period_high = period_mean - n_std * period_std, period_mean + n_std * period_std
            power_low, power_high = power_mean - n_std * power_std, power_mean + n_std * power_std
        else:
            wt_sub = sub[sub["type"] == "WT"]
            if len(wt_sub) > 1:
                period_mean, period_std = wt_sub["peak_period"].mean(), wt_sub["peak_period"].std()
                power_mean, power_std = wt_sub["peak_power"].mean(), wt_sub["peak_power"].std()
                period_low, period_high = period_mean - n_std * period_std, period_mean + n_std * period_std
                power_low, power_high = power_mean - n_std * power_std, power_mean + n_std * power_std
            else:
                period_low = period_high = power_low = power_high = None

        for idx, row in sub.iterrows():
            phenotypes = []
            if period_low is not None and not row.get("is_arrhythmic", False):
                if row["peak_period"] < period_low:
                    phenotypes.append("low period")
                if row["peak_period"] > period_high:
                    phenotypes.append("high period")
                if row["peak_power"] < power_low:
                    phenotypes.append("low amplitude")
                if row["peak_power"] > power_high:
                    phenotypes.append("high amplitude")
            if row.get("is_arrhythmic", False):
                phenotypes.append("arrhythmic")
            df.at[idx, "phenotype"] = ", ".join(phenotypes)
    return df

def build_phenotype_table(all_gene_results, custom_wt_stats=None):
    labeled = label_phenotypes(all_gene_results, custom_wt_stats)
    if labeled.empty:
        return {}
    tables = {}
    out_columns = ["plate_id", "run_nr", "strain", "type", "peak_period", "peak_power", "peak_to_mean_ratio", "arrhythmic_threshold", "phenotype"]
    out_columns = [c for c in out_columns if c in labeled.columns]
    for ds in labeled["dataset"].unique():
        sub = labeled[(labeled["dataset"] == ds) & (labeled["phenotype"] != "")]
        tables[ds] = sub[out_columns].reset_index(drop=True)
    return tables

def build_phenotype_summary(all_gene_results, custom_wt_stats=None):
    labeled = label_phenotypes(all_gene_results, custom_wt_stats)
    if labeled.empty:
        return pd.DataFrame()
    idx_cols = ["plate_id", "run_nr", "strain"] if "run_nr" in labeled.columns and "plate_id" in labeled.columns else ("run_nr" if "run_nr" in labeled.columns else "strain")
    summary = labeled.pivot_table(index=idx_cols, columns="dataset", values="phenotype", aggfunc="first")
    summary = summary.fillna("")
    type_map = labeled.drop_duplicates(idx_cols if isinstance(idx_cols, list) else [idx_cols]).set_index(idx_cols)["type"]
    summary.insert(0, "type", type_map)
    return summary.reset_index()

def save_phenotypes_to_excel(all_results, output_dir, filename="Phenotypes.xlsx", custom_wt_stats=None):
    if not all_results:
        return
    detail_tables = build_phenotype_table(all_results, custom_wt_stats)
    summary_table = build_phenotype_summary(all_results, custom_wt_stats)
    excel_path = os.path.join(output_dir, filename)
    total_flagged = sum(len(t) for t in detail_tables.values())
    os.makedirs(output_dir, exist_ok=True)
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        if not summary_table.empty:
            summary_table.to_excel(writer, sheet_name="Summary", index=False)
        for dataset, sub in detail_tables.items():
            sheet_name = dataset[:31]
            sub.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Saved phenotype file to: {excel_path} ({total_flagged} strain(s) flagged)")

# =============================================================================
# DATA PROCESSING FUNCTION
# =============================================================================

def process_dataset(csv_path, dataset_name, data_type, wt_identifiers, output_dir):
    print(f"  > Processing {dataset_name} ({data_type})")
    df = pd.read_csv(csv_path, sep=CONFIG['sep'], decimal=CONFIG['decimal'], header=[0, 1])
    time_col = None
    for col in df.columns:
        if any('time' in str(c).lower() for c in col):
            time_col = col
            break
    if time_col is None:
        time_col = df.columns[0]

    df = df.dropna(subset=[time_col]).reset_index(drop=True)
    raw_time = pd.to_numeric(df[time_col], errors='coerce').values * CONFIG['sampling_rate']
    mask = (raw_time >= CONFIG['t_min']) & (raw_time <= CONFIG['t_max'])
    time = raw_time[mask]
    df_filtered = df.loc[mask]

    results = []
    wt_raw_list, wt_rate_list, wt_det_list, wt_periods_list, wt_gws_list, wt_traces_grid = [], [], [], [], [], []

    well_cols = [c for c in df_filtered.columns if c != time_col]
    num_grid_wells = min(96, len(well_cols))
    all_col_names = [None] * 96
    well_traces = [[] for _ in range(96)]

    for i in range(num_grid_wells):
        col_tuple = well_cols[i]
        strain_name = str(col_tuple[1]).strip()
        all_col_names[i] = strain_name

    strain_to_well_idx = {all_col_names[i]: i for i in range(num_grid_wells) if all_col_names[i] is not None}
    skipped_cols = []

    for i, col_tuple in enumerate(well_cols):
        run_nr = col_tuple[0]
        strain_name = col_tuple[1]
        col_display = str(strain_name).strip()
        signal = pd.to_numeric(df_filtered[col_tuple], errors='coerce').fillna(0).values.astype(float)
        is_wt = any(tag in str(strain_name).lower() for tag in wt_identifiers)

        valid_idx = np.where(signal != 0)[0]
        if valid_idx.size == 0:
            skipped_cols.append(col_display)
            continue

        first_idx = valid_idx[0]
        last_idx = valid_idx[-1]
        time_trunc = time[first_idx:last_idx + 1]
        signal_trunc = signal[first_idx:last_idx + 1]

        if len(time_trunc) < 2 or len(np.unique(signal_trunc)) <= 1:
            skipped_cols.append(col_display)
            continue

        try:
            # All datasets (CAE, MCBI, MEBI) now use the Butterworth filter
            raw_val = signal_trunc
            trend_val, detrended_val, rate_val, trend_rate = filter_biolum_trend(time_trunc, signal_trunc)

            periods, power, gws = compute_cwt(time_trunc, detrended_val)
            features = extract_wavelet_features(periods, power, gws)

            features.update({
                "dataset": dataset_name,
                "strain": col_display,
                "run_nr": run_nr,
                "type": "WT" if is_wt else "Deletion",
                "t0_h": time_trunc[0],
                "frames_dropped": first_idx,
                "frames_dropped_end": len(signal) - 1 - last_idx
            })

            results.append(features)
            detrended_full = np.full(len(time), np.nan)
            detrended_full[first_idx:last_idx + 1] = detrended_val

            if i < 96:
                well_traces[i].append((detrended_full, is_wt, col_display, run_nr))
            else:
                if col_display in strain_to_well_idx:
                    target_well = strain_to_well_idx[col_display]
                    well_traces[target_well].append((detrended_full, is_wt, col_display, run_nr))

            if not is_wt:
                save_path = os.path.join(output_dir, dataset_name.replace(' ', '_'), 'genes', f"{col_display}_run{run_nr}.png")
                plot_4panel(time_trunc, raw_val, trend_val, rate_val, trend_rate, detrended_val, periods, power, gws, features["peak_period"], f"{dataset_name} | {col_display} (Run: {run_nr})", data_type, save_path)
            else:
                raw_full = np.full(len(time), np.nan)
                raw_full[first_idx:last_idx + 1] = raw_val
                rate_full = np.full(len(time), np.nan)
                rate_full[first_idx:last_idx + 1] = rate_val
                wt_raw_list.append(raw_full)
                wt_rate_list.append(rate_full)
                wt_det_list.append(detrended_full)
                wt_periods_list.append(periods)
                wt_gws_list.append(gws)
                wt_traces_grid.append([(detrended_full, col_display, run_nr)])
        except Exception as e:
            print(f"    Error processing well {col_display}: {e}")
            skipped_cols.append(col_display)
            continue

    if skipped_cols:
        print(f"    Skipped (all-zero / missing in window): {', '.join(skipped_cols)}")

    if wt_raw_list:
        save_path = os.path.join(output_dir, dataset_name.replace(' ', '_'), 'WT', "WT_aggregate.png")
        plot_wt_aggregate(time, np.vstack(wt_raw_list), np.vstack(wt_rate_list), np.vstack(wt_det_list), wt_periods_list, wt_gws_list, f"{dataset_name} | WT Aggregate", data_type, save_path)

    all_valid_traces = [trace for traces in well_traces for trace, _, _, _ in traces if not np.all(np.isnan(trace))]
    if all_valid_traces:
        all_matrix = np.vstack(all_valid_traces)
        y_min, y_max = np.nanmin(all_matrix), np.nanmax(all_matrix)
        pad = 0.05 * (y_max - y_min) if y_max != y_min else 1.0
        ylim = (y_min - pad, y_max + pad) if not np.isnan(y_min) else None
        plate_dir = os.path.join(output_dir, dataset_name.replace(' ', '_'), 'plate')
        
        save_path_full = os.path.join(plate_dir, f"{dataset_name.replace(' ', '_')}_96Well_Grid.png")
        plot_plate_grid(time, well_traces, all_col_names, f"{dataset_name} | Detrended Signals (Replicates Overlaid)", save_path_full, ylim=ylim)

        save_path_wt = os.path.join(plate_dir, f"{dataset_name.replace(' ', '_')}_WT_96Well_Grid.png")
        plot_wt_plate_grid(time, wt_traces_grid, f"{dataset_name} | WT Detrended Signals", save_path_wt, ylim=ylim)

    return results

# =============================================================================
# TWO-PASS EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":
    base_dir = r"E:\Full Circadian Screen"
    output_dir_base = os.path.join(base_dir, "Results_PMR_Butterworth")
    
    cae_dir = os.path.join(base_dir, "CAE")
    mcbi_dir = os.path.join(base_dir, "MCBI")
    mebi_dir = os.path.join(base_dir, "MEBI")

    wt_tags = ("wt",)
    cae_files = glob.glob(os.path.join(cae_dir, "CAE_*.csv"))

    if not cae_files:
        print(f"No CAE CSV files found in {cae_dir}. Please check your directory.")
    else:
        # =====================================================================
        # PASS 1: Collect WT baselines and establish run-specific thresholds
        # =====================================================================
        print("==================================================================")
        print("                 PASS 1: WT BASELINE EXTRACTION")
        print("==================================================================")
        
        all_global_results = []
        
        for cae_path in cae_files:
            filename = os.path.basename(cae_path)
            plate_id = filename.replace("CAE_", "").replace(".csv", "")
            mcbi_path = os.path.join(mcbi_dir, f"MCBI_{plate_id}.csv")
            mebi_path = os.path.join(mebi_dir, f"MEBI_{plate_id}.csv")

            if not (os.path.exists(mcbi_path) and os.path.exists(mebi_path)):
                continue
                
            plate_output_dir = os.path.join(output_dir_base, f"Plate_{plate_id}")
            plate_results = []

            plate_results += process_dataset(cae_path, "Colony Area Expansion", "CAE", wt_tags, plate_output_dir)
            plate_results += process_dataset(mcbi_path, "Mean Colony Bioluminescence Intensity", "MCBI", wt_tags, plate_output_dir)
            plate_results += process_dataset(mebi_path, "Mean Edge Bioluminescence Intensity", "MEBI", wt_tags, plate_output_dir)

            for r in plate_results:
                r["plate_id"] = plate_id

            all_global_results.extend(plate_results)

        # Compute cross-plate run-specific thresholds and WT statistics from Pass 1 data
        df_all = pd.DataFrame(all_global_results)
        custom_wt_stats = {}

        for ds in df_all["dataset"].unique():
            sub_ds = df_all[df_all["dataset"] == ds]
            for run_val, run_sub in sub_ds.groupby("run_nr"):
                wt_pooled = run_sub[run_sub["type"] == "WT"]
                if not wt_pooled.empty:
                    arrhythmic_threshold = np.percentile(wt_pooled["peak_to_mean_ratio"].values, 5)
                    custom_wt_stats[(ds, run_val)] = (
                        wt_pooled["peak_period"].mean(),
                        wt_pooled["peak_period"].std(),
                        wt_pooled["peak_power"].mean(),
                        wt_pooled["peak_power"].std()
                    )
                else:
                    arrhythmic_threshold = None

                mask = (df_all["dataset"] == ds) & (df_all["run_nr"] == run_val)
                df_all.loc[mask, "arrhythmic_threshold"] = arrhythmic_threshold
                df_all.loc[mask, "is_arrhythmic"] = (arrhythmic_threshold is not None) & (df_all.loc[mask, "peak_to_mean_ratio"] < arrhythmic_threshold)

        all_global_results = df_all.to_dict(orient="records")

        # =====================================================================
        # PASS 2: Generate final plate-level and global plots/reports with WTs
        # =====================================================================
        print("\n==================================================================")
        print("           PASS 2: GENERATING REPORTS & PLOTS WITH WTS")
        print("==================================================================")

        for plate_id, plate_sub_df in df_all.groupby("plate_id"):
            plate_output_dir = os.path.join(output_dir_base, f"Plate_{plate_id}")
            plate_records = plate_sub_df.to_dict(orient="records")
            
            plot_clustering_separate(plate_records, plate_output_dir, custom_wt_stats=custom_wt_stats)
            plot_period_correlation(plate_records, plate_output_dir)
            save_results_to_excel(plate_records, plate_output_dir, "Plate_Rhythmicity_Results.xlsx")
            save_phenotypes_to_excel(plate_records, plate_output_dir, "Plate_Phenotypes.xlsx", custom_wt_stats=custom_wt_stats)
            print(f"Finished Plate {plate_id} analyses and figures.")

        # Global overview outputs
        global_plate_dir = os.path.join(output_dir_base, "Global_PlateBased")
        plot_clustering_separate(all_global_results, global_plate_dir, custom_wt_stats=custom_wt_stats)
        plot_period_correlation(all_global_results, global_plate_dir)
        save_results_to_excel(all_global_results, global_plate_dir, "Rhythmicity_Results_Global.xlsx")
        save_phenotypes_to_excel(all_global_results, global_plate_dir, "Phenotypes_Global.xlsx", custom_wt_stats=custom_wt_stats)

        print("\nScreening and reporting completely finished successfully.")