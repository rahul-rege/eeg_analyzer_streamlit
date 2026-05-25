"""Standalone single-file Streamlit UI for the EEG before/after dashboard.

This file is a self-contained merge of:

* ``src/eeg_v6_may22_stable.py`` — EEG reading, channel selection, PSD
  computation, band-power aggregation, and dashboard rendering.
* ``src/streamlit/app.py``       — Streamlit UI glue around v6.

There are no sibling-module imports, so this script can be shipped on
its own and launched with:

    streamlit run app_standalone.py

Two mutually-exclusive collapsible workflows are exposed on top of
``render_combined_dashboard`` / ``render_paired_dashboard``:

* **Person to Multiple Mudras** — one ``{person}_b4.eeg`` baseline +
  several ``{person}_af_{mudra}.eeg`` after files for the same person.
* **Multiple Persons Same Mudra** — N ``{person}_b4.eeg`` baselines + N
  ``{person}_af_{mudra}.eeg`` after files (all the same mudra), paired
  by person name.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import streamlit as st
from matplotlib.figure import Figure
from matplotlib.patches import Patch


# ======================================================================
# Analysis pipeline — inlined from src/eeg_v6_may22_stable.py
# ======================================================================

# Default acquisition profile used by the Streamlit app. Twenty-one
# channels in the standard 10-20 montage.
DEFAULT_NUM_CHANNELS: int = 21
DEFAULT_SFREQ: float = 256.0
DEFAULT_BIT_TO_UV: float = 0.045
DEFAULT_CH_NAMES: tuple[str, ...] = (
    "Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz",
    "T3", "T4", "C3", "C4", "Cz",
    "A1", "A2",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "O2",
)


SYSTEM_CONFIGS = {
    'Default_System': {
        'NUM_CHANNELS': 10,
        'DEFAULT_FS': 256,
        'BIT_TO_UV': 0.045,
        'CHANNEL_NAMES': [f'EEG {i+1:03d}' for i in range(10)]
    },
    'NEUROSET_NR_1001': {
        'NUM_CHANNELS': None,
        'DEFAULT_FS': None,
        'BIT_TO_UV': None,
        'CHANNEL_NAMES': None
    },
    'Clarity_Medical_10_20': {
        'NUM_CHANNELS': None,
        'DEFAULT_FS': None,
        'BIT_TO_UV': None,
        'CHANNEL_NAMES': None
    }
}


# Gamma band is intentionally excluded — matches the Colab notebook.
freq_bands = {
    'Delta': [0.5, 4],
    'Theta': [4, 8],
    'Alpha': [8, 13],
    'Beta':  [13, 30],
}


def load_raw_binary_eeg(file_path, n_channels, sfreq, bit_to_uv, ch_names):
    """Reads raw binary EEG data and creates an MNE RawArray object."""
    try:
        with open(file_path, 'rb') as f:
            data_bytes = f.read()

        eeg_data = np.frombuffer(data_bytes, dtype='<i2')

        n_times = len(eeg_data) // n_channels
        if len(eeg_data) % n_channels != 0:
            print(f"Warning: Data length ({len(eeg_data)}) not perfectly divisible by number of channels ({n_channels}). Truncating last incomplete sample.")
            eeg_data = eeg_data[:n_times * n_channels]

        eeg_data_uv = eeg_data.reshape(n_channels, n_times) * bit_to_uv
        eeg_data_volts = eeg_data_uv * 1e-6

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

        raw = mne.io.RawArray(eeg_data_volts, info, verbose=False)
        return raw

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading binary EEG file {file_path}: {e}")


def select_channels(raw, selected_channels='all'):
    """Selects EEG channels from a raw MNE object."""
    if selected_channels == 'all':
        picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
        channels_to_analyze = [raw.ch_names[i] for i in picks]
        print(f"Analyzing all {len(channels_to_analyze)} EEG channels.")
    else:
        available_eeg_channels = mne.pick_types(raw.info, eeg=True, exclude='bads')
        available_eeg_channel_names = [raw.ch_names[i] for i in available_eeg_channels]

        channels_to_analyze = [ch for ch in selected_channels if ch in available_eeg_channel_names]
        if not channels_to_analyze:
            print("None of the selected channels were found in the EEG data. Please check channel names.")
            print("Available EEG channels:", available_eeg_channel_names)
            return None
        else:
            print(f"Analyzing selected EEG channels: {channels_to_analyze}")

    return raw.copy().pick_channels(channels_to_analyze)


def plot_psd(raw_selected, title_suffix="", color='mediumblue', ax=None):
    """Calculates and plots the PSD for the selected channels on a given Axes object."""
    if ax is None:
        ax = plt.gca()

    if raw_selected:
        spectrum = raw_selected.compute_psd(method='welch', fmin=0.5, fmax=100.,
                                            picks='eeg', n_fft=2048, n_per_seg=None,
                                            verbose=False)
        psds_per_channel_uv = spectrum.get_data(fmin=0.5, fmax=100.) * (1e6)**2
        psds_mean_across_channels_uv = psds_per_channel_uv.mean(axis=0)

        freqs = spectrum.freqs

        ax.plot(freqs, psds_mean_across_channels_uv, color=color, linewidth=2, label=title_suffix)
    else:
        print(f"Cannot plot PSD: No raw data provided for {title_suffix}")


def calculate_band_power(raw_selected, freq_bands):
    """Calculates average power in specified frequency bands for selected channels.
    Returns a dictionary of band powers in uV^2.
    """
    band_powers = {}
    if raw_selected is None:
        return band_powers

    spectrum = raw_selected.compute_psd(method='welch', fmin=0.5, fmax=100.,
                                        picks='eeg', n_fft=2048, n_per_seg=None,
                                        average='mean', verbose=False)

    psds_per_channel = spectrum.get_data(fmin=0.5, fmax=100.)
    freqs_psd = spectrum.freqs

    psds_per_channel_uv = psds_per_channel * (1e6)**2

    for band_name, (fmin, fmax) in freq_bands.items():
        idx_band = np.logical_and(freqs_psd >= fmin, freqs_psd <= fmax)

        if np.any(idx_band):
            band_psds = psds_per_channel_uv[:, idx_band]

            freq_res = freqs_psd[1] - freqs_psd[0]
            power_in_band_per_channel = np.sum(band_psds, axis=1) * freq_res
            band_powers[band_name] = np.mean(power_in_band_per_channel)
        else:
            band_powers[band_name] = 0.0

    return band_powers


def load_and_prepare_eeg(
    file_path,
    n_channels,
    sfreq,
    bit_to_uv,
    ch_names,
    selected_channels='all',
):
    """Load a `.eeg` binary, apply CAR + bandpass, then restrict channels."""
    raw = load_raw_binary_eeg(file_path, n_channels, sfreq, bit_to_uv, ch_names)

    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    if len(eeg_picks) > 1:
        raw.set_eeg_reference(ref_channels='average', projection=True, ch_type='eeg')
        raw.apply_proj()

    raw.filter(l_freq=0.1, h_freq=100., picks='eeg', fir_design='firwin')

    return select_channels(raw, selected_channels)


def render_combined_dashboard(
    raw_selected_data: dict,
    freq_bands: dict = freq_bands,
    suptitle: str = "",
) -> Figure:
    """Reproduce the notebook's combined PSD + band-power histogram figure."""
    condition_names = [name for name, raw in raw_selected_data.items() if raw]
    n_conditions = len(condition_names)
    if n_conditions == 0:
        raise ValueError("No valid recordings supplied to render the dashboard.")

    cmap = plt.colormaps.get_cmap('viridis')
    colors_list = [cmap(x) for x in np.linspace(0, 1, n_conditions)]
    plot_colors = {name: colors_list[i] for i, name in enumerate(condition_names)}

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax1 = axes[0]
    for name in condition_names:
        plot_psd(raw_selected_data[name], title_suffix=name,
                 color=plot_colors[name], ax=ax1)
    ax1.set_xlabel('Frequency (Hz)', fontsize=12)
    ax1.set_ylabel('Power Spectral Density ($µV^2/Hz$)', fontsize=12)
    ax1.set_title('Comparative Average Power Spectral Density', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xlim(0.5, 100)
    ax1.set_yscale('log')
    ax1.legend()

    ax2 = axes[1]
    all_band_powers = {
        name: calculate_band_power(raw_selected_data[name], freq_bands)
        for name in condition_names
    }
    band_names = list(freq_bands.keys())
    n_bands = len(band_names)
    bar_width = 0.8 / n_conditions
    index = np.arange(n_bands)

    for i, name in enumerate(condition_names):
        values = [all_band_powers[name][b] for b in band_names]
        ax2.bar(index + i * bar_width, values, bar_width, label=name,
                color=plot_colors[name])

    ax2.set_xlabel('Frequency Band', fontsize=12)
    ax2.set_ylabel('Average Power ($µV^2$)', fontsize=12)
    ax2.set_title('Comparative Average Power in EEG Frequency Bands', fontsize=14)
    ax2.set_xticks(index + bar_width * (n_conditions - 1) / 2, band_names)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    ax2.legend(title='Condition')

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, fontweight='bold')
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()
    return fig


def render_paired_dashboard(
    paired_data,
    freq_bands: dict = freq_bands,
    suptitle: str = "",
) -> Figure:
    """Render a Before-vs-After comparison for N persons sharing a mudra."""
    BEFORE_COLOR = "#FF6347"
    AFTER_COLOR = "#4682B4"

    pairs = [(p, b, a) for p, b, a in paired_data if b is not None and a is not None]
    if not pairs:
        raise ValueError("No valid person pairs supplied to render_paired_dashboard.")
    n_persons = len(pairs)

    cmap_name = 'tab10' if n_persons <= 10 else 'tab20'
    cmap = plt.colormaps.get_cmap(cmap_name)
    person_color = {p: cmap(i % cmap.N) for i, (p, _, _) in enumerate(pairs)}

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    ax1, ax2 = axes

    for person, before_raw, after_raw in pairs:
        color = person_color[person]
        for label_suffix, raw, ls in (
            (' (Before)', before_raw, '-'),
            (' (After)', after_raw, '--'),
        ):
            spectrum = raw.compute_psd(
                method='welch', fmin=0.5, fmax=100.,
                picks='eeg', n_fft=2048, n_per_seg=None, verbose=False,
            )
            psd_uv = spectrum.get_data(fmin=0.5, fmax=100.) * (1e6) ** 2
            ax1.plot(
                spectrum.freqs, psd_uv.mean(axis=0),
                color=color, linewidth=2, linestyle=ls,
                label=f"{person}{label_suffix}",
            )

    ax1.set_xlabel('Frequency (Hz)', fontsize=12)
    ax1.set_ylabel('Power Spectral Density ($µV^2/Hz$)', fontsize=12)
    ax1.set_title('Comparative Power Spectral Density', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xlim(0.5, 100)
    ax1.set_yscale('log')
    ax1.legend(fontsize=9)

    band_names = list(freq_bands.keys())
    n_bands = len(band_names)
    band_x = np.arange(n_bands)

    group_width = 0.85
    cell_width = group_width / n_persons
    bar_width = cell_width * 0.45

    powers_cache = {
        p: (calculate_band_power(b, freq_bands),
            calculate_band_power(a, freq_bands))
        for p, b, a in pairs
    }

    for i, (person, _, _) in enumerate(pairs):
        cell_offset = (i - (n_persons - 1) / 2) * cell_width
        before_x = band_x + cell_offset - bar_width / 2
        after_x = band_x + cell_offset + bar_width / 2

        before_vals = [powers_cache[person][0][b] for b in band_names]
        after_vals = [powers_cache[person][1][b] for b in band_names]

        ax2.bar(
            before_x, before_vals, bar_width,
            color=BEFORE_COLOR,
            edgecolor=person_color[person], linewidth=1.4,
        )
        ax2.bar(
            after_x, after_vals, bar_width,
            color=AFTER_COLOR,
            edgecolor=person_color[person], linewidth=1.4,
        )

    ax2.set_xticks(band_x, band_names)
    ax2.set_xlabel('Frequency Band', fontsize=12)
    ax2.set_ylabel('Average Power ($µV^2$)', fontsize=12)
    ax2.set_title('Per-Person Band Power: Before vs After', fontsize=14)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    legend_handles = [
        Patch(facecolor=BEFORE_COLOR, edgecolor='black', label='Before'),
        Patch(facecolor=AFTER_COLOR, edgecolor='black', label='After'),
    ]
    legend_handles += [
        Patch(facecolor='white', edgecolor=person_color[p], linewidth=1.6, label=p)
        for p, _, _ in pairs
    ]
    ax2.legend(handles=legend_handles, fontsize=9, loc='upper right',
               title='Phase / Person')

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, fontweight='bold')
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()
    return fig


# Alias so the UI code below can keep using its original `v6.xxx` references
# without renaming.
v6 = sys.modules[__name__]


# ======================================================================
# Streamlit UI — inlined from src/streamlit/app.py
# ======================================================================

st.set_page_config(
    page_title="EEG Before/After Dashboard",
    page_icon="🧠",
    layout="wide",
)


# ----------------------------------------------------------------------
# Generic helpers
# ----------------------------------------------------------------------

def _persist_upload(uploaded_file, suffix: str = ".eeg") -> Path:
    """Write a Streamlit UploadedFile to a temp path so v6 can open it."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getvalue())
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def _load_one(uploaded_file, cfg: dict):
    """Persist a single uploaded `.eeg` and run it through v6's pipeline."""
    path = _persist_upload(uploaded_file)
    return v6.load_and_prepare_eeg(
        str(path),
        n_channels=cfg["num_channels"],
        sfreq=cfg["sfreq"],
        bit_to_uv=cfg["bit_to_uv"],
        ch_names=cfg["ch_names"],
        selected_channels=cfg["picked"],
    )


# ----------------------------------------------------------------------
# Filename parsing — convention:
#   Before : {person}_b4.eeg               (e.g. Shreya_b4.eeg)
#   After  : {person}_af_{mudra}.eeg       (e.g. Shreya_af_hakini.eeg)
# Matching is case-insensitive; mudra/person can be multi-token.
# ----------------------------------------------------------------------

def _parse_before_name(filename: str) -> str:
    """Return the person name parsed from a `{person}_b4.eeg` style file."""
    stem = Path(filename).stem
    parts = stem.split("_")
    kept = [p for p in parts if p.lower() != "b4"]
    return ("_".join(kept) if kept else stem) or "Unknown"


def _parse_after_name(filename: str) -> tuple[str, str]:
    """Return (person, mudra) from a `{person}_af_{mudra}.eeg` file."""
    stem = Path(filename).stem
    parts = stem.split("_")
    lower = [p.lower() for p in parts]
    if "af" in lower:
        idx = lower.index("af")
        person = "_".join(parts[:idx]) or "Unknown"
        mudra = "_".join(parts[idx + 1:]) or "Unknown"
        return person, mudra
    if len(parts) >= 2:
        return "_".join(parts[:-1]), parts[-1]
    return stem, "Unknown"


def _unique_label(base: str, existing: dict) -> str:
    """Disambiguate `base` against keys already in `existing` by appending (n)."""
    if base not in existing:
        return base
    n = 2
    while f"{base} ({n})" in existing:
        n += 1
    return f"{base} ({n})"


# ----------------------------------------------------------------------
# Sidebar: acquisition settings + per-channel checkboxes
# ----------------------------------------------------------------------

def _channel_selector(num_channels: int) -> tuple[list[str], list[str]]:
    st.markdown("**Channels**")
    st.caption(
        "Names define how the interleaved binary maps to electrodes. "
        "Uncheck a row to exclude that channel from the analysis."
    )

    name_col, keep_col = st.columns([3, 1])
    name_col.markdown("_Name_")
    keep_col.markdown("_Use_")

    names: list[str] = []
    picked: list[str] = []
    for i in range(num_channels):
        default_name = (
            v6.DEFAULT_CH_NAMES[i] if i < len(v6.DEFAULT_CH_NAMES) else f"Ch{i + 1}"
        )
        with name_col:
            name = st.text_input(
                f"Channel {i + 1} name",
                value=default_name,
                key=f"ch_name_{i}",
                label_visibility="collapsed",
            )
        with keep_col:
            keep = st.checkbox(
                f"Use channel {i + 1}",
                value=True,
                key=f"ch_keep_{i}",
                label_visibility="collapsed",
            )
        name = name.strip() or default_name
        names.append(name)
        if keep:
            picked.append(name)
    return names, picked


def _sidebar_config() -> dict:
    st.sidebar.header("Acquisition settings")
    num_channels = st.sidebar.number_input(
        "Channels in file",
        min_value=1,
        max_value=64,
        value=v6.DEFAULT_NUM_CHANNELS,
        step=1,
        help="How many channels are interleaved in the raw binary. "
             "Must match the recording rig.",
    )
    sfreq = st.sidebar.number_input(
        "Sampling rate (Hz)",
        min_value=1.0,
        max_value=10_000.0,
        value=float(v6.DEFAULT_SFREQ),
        step=1.0,
    )
    bit_to_uv = st.sidebar.number_input(
        "ADC scale (µV per count)",
        min_value=1e-6,
        max_value=1e3,
        value=float(v6.DEFAULT_BIT_TO_UV),
        step=0.001,
        format="%.6f",
    )

    st.sidebar.divider()
    with st.sidebar:
        ch_names, picked = _channel_selector(int(num_channels))

    return {
        "num_channels": int(num_channels),
        "sfreq": float(sfreq),
        "bit_to_uv": float(bit_to_uv),
        "ch_names": ch_names,
        "picked": picked,
    }


# ----------------------------------------------------------------------
# Section builders
# ----------------------------------------------------------------------

def _build_section1_payload(
    before_file,
    after_files: list,
    cfg: dict,
) -> tuple[dict, str]:
    """Section 1: one person, one baseline, multiple mudras."""
    person = _parse_before_name(before_file.name)
    raw_data: dict = {}
    raw_data["Before"] = _load_one(before_file, cfg)

    skipped: list[str] = []
    for af in after_files:
        af_person, mudra = _parse_after_name(af.name)
        if af_person.lower() != person.lower():
            skipped.append(f"`{af.name}` (parsed person {af_person!r})")
            continue
        label = _unique_label(mudra.capitalize() or "Mudra", raw_data)
        raw_data[label] = _load_one(af, cfg)

    if skipped:
        st.warning(
            f"Section 1 expects all files for the same person ({person!r}). "
            f"Skipped: {', '.join(skipped)}"
        )

    suptitle = f"Person: {person} — Before vs Mudras"
    return raw_data, suptitle


def _build_section2_payload(
    before_files: list,
    after_files: list,
    cfg: dict,
) -> tuple[list, str]:
    """Section 2: multiple persons, one mudra, paired before/after per person."""
    before_by_key: dict = {}
    for bf in before_files:
        person = _parse_before_name(bf.name)
        before_by_key.setdefault(person.lower(), (person, bf))

    after_by_key: dict = {}
    mudras_seen: list[str] = []
    for af in after_files:
        person, mudra = _parse_after_name(af.name)
        after_by_key.setdefault(person.lower(), (person, af, mudra))
        if mudra and mudra not in mudras_seen:
            mudras_seen.append(mudra)

    if len(mudras_seen) > 1:
        st.warning(
            "Section 2 expects all After files to share the same mudra, "
            f"but found: {', '.join(mudras_seen)}. Using {mudras_seen[0]!r} for the title."
        )
    mudra_name = (mudras_seen[0] if mudras_seen else "Unknown").capitalize()

    matched_keys = sorted(set(before_by_key) & set(after_by_key))
    unmatched_before = sorted(set(before_by_key) - set(after_by_key))
    unmatched_after = sorted(set(after_by_key) - set(before_by_key))
    if unmatched_before or unmatched_after:
        msg_parts = []
        if unmatched_before:
            msg_parts.append(
                "before-only: " + ", ".join(before_by_key[k][0] for k in unmatched_before)
            )
        if unmatched_after:
            msg_parts.append(
                "after-only: " + ", ".join(after_by_key[k][0] for k in unmatched_after)
            )
        st.warning(
            "Section 2 pairs files by person name; unpaired files were skipped "
            "(" + "; ".join(msg_parts) + ")."
        )

    paired_data: list = []
    for key in matched_keys:
        bf_name, bf_file = before_by_key[key]
        _af_name, af_file, _mudra = after_by_key[key]
        paired_data.append((bf_name, _load_one(bf_file, cfg), _load_one(af_file, cfg)))

    suptitle = f"Mudra: {mudra_name} — {len(paired_data)} person(s)"
    return paired_data, suptitle


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------

def _show_section1_dashboard(raw_selected_data: dict, suptitle: str) -> None:
    """Render Section 1's dashboard (one person, multiple mudras)."""
    if len(raw_selected_data) < 2:
        st.error("Need a Before plus at least one After to render a Section 1 dashboard.")
        return

    fig = v6.render_combined_dashboard(
        raw_selected_data,
        freq_bands=v6.freq_bands,
        suptitle=suptitle,
    )

    st.subheader("Dashboard")
    st.pyplot(fig, use_container_width=True)

    summary_rows = []
    for label, raw in raw_selected_data.items():
        if raw is None:
            continue
        powers = v6.calculate_band_power(raw, v6.freq_bands)
        summary_rows.append({"Condition": label, **powers})

    st.subheader("Band-power summary (µV²)")
    st.dataframe(summary_rows, use_container_width=True)


def _show_section2_dashboard(paired_data: list, suptitle: str) -> None:
    """Render Section 2's dashboard (multiple persons, paired before/after)."""
    valid_pairs = [(p, b, a) for p, b, a in paired_data if b is not None and a is not None]
    if not valid_pairs:
        st.error("Need at least one fully-paired person (Before + After) to render.")
        return

    fig = v6.render_paired_dashboard(
        valid_pairs,
        freq_bands=v6.freq_bands,
        suptitle=suptitle,
    )

    st.subheader("Dashboard")
    st.pyplot(fig, use_container_width=True)

    summary_rows = []
    for person, before_raw, after_raw in valid_pairs:
        b_pow = v6.calculate_band_power(before_raw, v6.freq_bands)
        a_pow = v6.calculate_band_power(after_raw, v6.freq_bands)
        summary_rows.append({"Person": person, "Phase": "Before", **b_pow})
        summary_rows.append({"Person": person, "Phase": "After", **a_pow})

    st.subheader("Band-power summary (µV²)")
    st.dataframe(summary_rows, use_container_width=True)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    st.title("🧠 EEG Before/After Dashboard")
    st.markdown(
        "Pick **one** of the two workflows below, upload the matching "
        "`.eeg` files, and click **Generate dashboard**."
    )
    st.caption(
        "Naming convention: Before files are `{person}_b4.eeg`; After files "
        "are `{person}_af_{mudra}.eeg`. e.g. `Shreya_b4.eeg`, "
        "`Shreya_af_hakini.eeg`."
    )

    cfg = _sidebar_config()

    with st.expander(
        "Section 1 — Person to Multiple Mudras (1 Before + N After)",
        expanded=True,
    ):
        st.caption(
            "Single baseline for one person plus one After file per mudra. "
            "All filenames should start with the same person name."
        )
        s1_before = st.file_uploader(
            "Before file (`{person}_b4.eeg`)",
            type=["eeg"],
            key="s1_before",
        )
        s1_afters = st.file_uploader(
            "After files (one per mudra, `{person}_af_{mudra}.eeg`)",
            type=["eeg"],
            accept_multiple_files=True,
            key="s1_afters",
        )
        if s1_before or s1_afters:
            before_person = _parse_before_name(s1_before.name) if s1_before else "—"
            after_pairs = [_parse_after_name(f.name) for f in (s1_afters or [])]
            after_summary = ", ".join(
                f"{p}/{m}" for p, m in after_pairs
            ) or "—"
            st.markdown(
                f"**Parsed** — Before person: `{before_person}` · "
                f"After person/mudra: `{after_summary}`"
            )

    with st.expander(
        "Section 2 — Multiple Persons Same Mudra (N Before + N After)",
        expanded=False,
    ):
        st.caption(
            "Equal-size lists of Before/After files paired by person name "
            "(case-insensitive). All After files must reference the same mudra."
        )
        s2_befores = st.file_uploader(
            "Before files (one per person)",
            type=["eeg"],
            accept_multiple_files=True,
            key="s2_befores",
        )
        s2_afters = st.file_uploader(
            "After files (same mudra, one per person)",
            type=["eeg"],
            accept_multiple_files=True,
            key="s2_afters",
        )
        if s2_befores or s2_afters:
            before_persons = [_parse_before_name(f.name) for f in (s2_befores or [])]
            after_pairs = [_parse_after_name(f.name) for f in (s2_afters or [])]
            before_keys = {p.lower() for p in before_persons}
            after_keys = {p.lower() for p, _ in after_pairs}
            paired = sorted(before_keys & after_keys)
            st.markdown(
                f"**Parsed** — Before persons: `{', '.join(before_persons) or '—'}` · "
                f"After person/mudra: "
                f"`{', '.join(f'{p}/{m}' for p, m in after_pairs) or '—'}` · "
                f"**Will pair {len(paired)} person(s)**: "
                f"`{', '.join(paired) or '—'}`"
            )

    st.divider()

    s1_files = bool(s1_before) or bool(s1_afters)
    s2_files = bool(s2_befores) or bool(s2_afters)
    s1_ready = bool(s1_before) and bool(s1_afters)
    s2_ready = bool(s2_befores) and bool(s2_afters) and len(s2_befores) == len(s2_afters)

    if s1_files and s2_files:
        st.error(
            "Files are present in both sections. Only one section can be used "
            "at a time — clear the section you don't want."
        )
    elif not (s1_files or s2_files):
        st.info("Upload files in one section to enable the dashboard.")
    else:
        if s1_files and not s1_ready:
            st.warning(
                "Section 1 needs a single Before file and at least one After file."
            )
        if s2_files and not s2_ready:
            if not s2_befores or not s2_afters:
                st.warning("Section 2 needs at least one Before and one After file.")
            elif len(s2_befores) != len(s2_afters):
                st.warning(
                    f"Section 2 needs the same number of Before and After files "
                    f"(got {len(s2_befores)} Before / {len(s2_afters)} After)."
                )

    has_channels = len(cfg["picked"]) > 0
    name_count_ok = len(cfg["ch_names"]) == cfg["num_channels"]
    unique_names = len(set(cfg["ch_names"])) == len(cfg["ch_names"])
    if not has_channels:
        st.warning("Select at least one channel to analyze.")
    if not unique_names:
        st.warning("Channel names must be unique.")
    if not name_count_ok:
        st.warning("Channel name count doesn't match the channel-in-file count.")

    exactly_one_section = (s1_ready and not s2_files) or (s2_ready and not s1_files)
    can_run = (
        exactly_one_section
        and has_channels
        and unique_names
        and name_count_ok
    )

    if st.button("Generate dashboard", type="primary", disabled=not can_run):
        try:
            if s1_ready:
                raw_data, suptitle = _build_section1_payload(s1_before, s1_afters, cfg)
                _show_section1_dashboard(raw_data, suptitle)
            else:
                paired_data, suptitle = _build_section2_payload(s2_befores, s2_afters, cfg)
                _show_section2_dashboard(paired_data, suptitle)
        except Exception as exc:
            st.error(f"Failed to render dashboard: {exc}")
            st.exception(exc)


if __name__ == "__main__":
    main()
