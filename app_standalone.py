"""Standalone single-file Streamlit UI for the EEG before/after dashboard.

This file is a self-contained merge of:

* ``src/eeg_v6_may22_stable.py`` — EEG reading, channel selection, PSD
  computation, band-power aggregation, and dashboard rendering.
* ``src/streamlit/app.py``       — Streamlit UI glue around v6.

There are no sibling-module imports, so this script can be shipped on
its own and launched with:

    streamlit run app_standalone.py

A single file uploader accepts every ``.eeg`` file at once. Each file
follows the loose convention ``{Person}_{state}_{Mudra}.eeg`` where
*state* is a before/after token (``b4``/``af`` and synonyms, any
casing). Files are paired into Before-vs-After comparisons on
``(person, mudra)`` and rendered with ``render_paired_dashboard``. A
single supplied baseline is reused for any mudra that lacks its own
before file.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure


def _fmt_power(value: float) -> str:
    """Readable plain-number label for a µV² band-power value.

    Values land in roughly 0.1–1000 µV², so prefer ordinary decimals over
    scientific notation (only the very tiny tail falls back to it).
    """
    if value is None or not np.isfinite(value):
        return "—"
    mag = abs(value)
    if mag == 0:
        return "0"
    if mag >= 100:
        return f"{value:,.0f}"
    if mag >= 1:
        return f"{value:.1f}"
    if mag >= 0.01:
        return f"{value:.2f}"
    return f"{value:.1e}"


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


def render_paired_dashboard(
    paired_data,
    freq_bands: dict = freq_bands,
    suptitle: str = "",
) -> Figure:
    """Render a Before-vs-After comparison across one or more recordings."""
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

    powers_cache = {
        label: (calculate_band_power(b, freq_bands),
                calculate_band_power(a, freq_bands))
        for label, b, a in pairs
    }

    # Each (recording, phase) is its own distinctly-coloured bar; bars are
    # grouped under their frequency band (Delta / Theta / Alpha / ...).
    single_pair = n_persons == 1
    series: list[tuple[str, list]] = []
    for label, _, _ in pairs:
        before_vals = [powers_cache[label][0][b] for b in band_names]
        after_vals = [powers_cache[label][1][b] for b in band_names]
        before_name = "Before" if single_pair else f"{label} — Before"
        after_name = "After" if single_pair else f"{label} — After"
        series.append((before_name, before_vals))
        series.append((after_name, after_vals))

    n_series = len(series)
    series_cmap = plt.colormaps.get_cmap('tab20' if n_series <= 20 else 'viridis')

    def _series_color(j: int):
        if n_series <= 20:
            return series_cmap(j % 20)
        return series_cmap(j / max(n_series - 1, 1))

    group_width = 0.85
    bar_width = group_width / n_series

    for j, (series_label, values) in enumerate(series):
        offset = (j - (n_series - 1) / 2) * bar_width
        bars = ax2.bar(
            band_x + offset, values, bar_width,
            label=series_label, color=_series_color(j),
        )
        ax2.bar_label(
            bars, labels=[_fmt_power(v) for v in values],
            rotation=90, padding=2, fontsize=6,
        )

    # Headroom so the rotated value labels above the tallest bars aren't clipped.
    ax2.margins(y=0.18)
    ax2.set_xticks(band_x, band_names)
    ax2.set_xlabel('Frequency Band', fontsize=12)
    ax2.set_ylabel('Average Power ($µV^2$)', fontsize=12)
    ax2.set_title('Band Power: Before vs After', fontsize=14)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    ax2.legend(fontsize=8, loc='upper right',
               title='Phase' if single_pair else 'Recording / Phase')

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
# Filename parsing — loose convention:
#   {PersonName}_{state}_{MudraName}.eeg
#
# `state` flags the recording phase and is matched case-insensitively
# against a small set of synonyms, so all of these are equivalent:
#   Before : Rahul_b4_Prana.eeg, Rahul_B4_prana.eeg, raHul_b4_prana.eeg
#   After  : Rahul_af_prana.eeg, Rahul_AF_prana.eeg
# Person and mudra may each span multiple `_`-separated tokens. A
# before/after pair is matched on (person, mudra), both lower-cased;
# when a mudra has no matching before, a single supplied baseline is
# reused as the comparison.
# ----------------------------------------------------------------------

BEFORE_TOKENS = {"b4", "before", "pre", "baseline", "base"}
AFTER_TOKENS = {"af", "after", "post"}


def _parse_file_name(filename: str) -> tuple[str, str | None, str]:
    """Parse `{person}_{state}_{mudra}.eeg` into (person, state, mudra).

    `state` is ``"before"``, ``"after"``, or ``None`` when no recognised
    phase token is present. Matching is case-insensitive.
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    lower = [p.lower() for p in parts]

    state = None
    state_idx = None
    for i, tok in enumerate(lower):
        if tok in BEFORE_TOKENS:
            state, state_idx = "before", i
            break
        if tok in AFTER_TOKENS:
            state, state_idx = "after", i
            break

    if state_idx is not None:
        person = "_".join(parts[:state_idx]) or "Unknown"
        mudra = "_".join(parts[state_idx + 1:]) or "Unknown"
    else:
        # No phase token found — best-effort: first token person, rest mudra.
        person = parts[0] if parts else stem
        mudra = "_".join(parts[1:]) if len(parts) > 1 else "Unknown"

    return person, state, mudra


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

def _build_payload(files: list, cfg: dict) -> tuple[list, str]:
    """Pair uploaded After files against Before baselines.

    Files are parsed with :func:`_parse_file_name`. Each After is matched
    to a Before on the ``(person, mudra)`` key (case-insensitive). When a
    mudra has no exact before, a baseline is reused as a fallback: the
    person's single before if they have exactly one, otherwise the single
    before supplied across all files. Unrecognised or unmatchable files
    produce a Streamlit warning and are skipped.
    """
    befores: list[dict] = []
    afters: list[dict] = []
    unparsed: list[str] = []
    for f in files:
        person, state, mudra = _parse_file_name(f.name)
        if state is None:
            unparsed.append(f.name)
            continue
        rec = {"person": person, "mudra": mudra, "file": f}
        (befores if state == "before" else afters).append(rec)

    if unparsed:
        st.warning(
            "Could not find a before/after token (b4/af) in: "
            + ", ".join(f"`{n}`" for n in unparsed)
            + ". Expected `{Person}_{state}_{Mudra}.eeg`."
        )

    # Before lookups: exact (person, mudra), per-person, and global single.
    before_by_pm: dict = {}
    before_by_person: dict = {}
    for b in befores:
        before_by_pm.setdefault((b["person"].lower(), b["mudra"].lower()), b)
        before_by_person.setdefault(b["person"].lower(), []).append(b)
    global_before = befores[0] if len(befores) == 1 else None

    # Cache loaded raws so a reused baseline is only processed once.
    _loaded: dict = {}

    def _load_rec(rec):
        fid = id(rec["file"])
        if fid not in _loaded:
            _loaded[fid] = _load_one(rec["file"], cfg)
        return _loaded[fid]

    def _resolve_before(after_rec):
        pk = (after_rec["person"].lower(), after_rec["mudra"].lower())
        if pk in before_by_pm:
            return before_by_pm[pk], False
        person_befores = before_by_person.get(after_rec["person"].lower(), [])
        if len(person_befores) == 1:
            return person_befores[0], True
        if global_before is not None:
            return global_before, True
        return None, False

    paired: list = []  # (person, mudra, before_rec, after_rec, is_fallback)
    unmatched: list[str] = []
    for a in afters:
        before_rec, is_fallback = _resolve_before(a)
        if before_rec is None:
            unmatched.append(f"{a['person']}/{a['mudra']}")
            continue
        paired.append((a["person"], a["mudra"], before_rec, a, is_fallback))

    if unmatched:
        st.warning(
            "No baseline available for: " + "; ".join(unmatched)
            + ". Upload a matching before file, or a single before file to "
            "reuse as the baseline."
        )

    fallbacks = [f"{p}/{m}" for p, m, _, _, fb in paired if fb]
    if fallbacks:
        st.info(
            "Reused a single baseline (no exact before file) for: "
            + ", ".join(fallbacks)
        )

    persons = {p for p, _, _, _, _ in paired}
    mudras = {m for _, m, _, _, _ in paired}
    multi_person = len(persons) > 1
    multi_mudra = len(mudras) > 1

    paired_data: list = []
    for person, mudra, before_rec, after_rec, _ in paired:
        if multi_person and multi_mudra:
            label = f"{person} — {mudra}"
        elif multi_mudra:
            label = mudra.capitalize()
        else:
            label = person
        paired_data.append((label, _load_rec(before_rec), _load_rec(after_rec)))

    if multi_person and multi_mudra:
        suptitle = f"Before vs After — {len(paired_data)} recording(s)"
    elif multi_mudra:
        suptitle = f"{next(iter(persons))} — Before vs After across mudras"
    elif multi_person:
        suptitle = f"{next(iter(mudras)).capitalize()} — {len(paired_data)} person(s)"
    elif paired_data:
        person, mudra = paired[0][0], paired[0][1]
        suptitle = f"{person} — {mudra.capitalize()}: Before vs After"
    else:
        suptitle = "Before vs After"

    return paired_data, suptitle


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------

def _build_band_power_table(paired_data: list) -> pd.DataFrame:
    """Long-format before/after band-power table mirroring the bar chart."""
    rows = []
    for label, before_raw, after_raw in paired_data:
        b_pow = v6.calculate_band_power(before_raw, v6.freq_bands)
        a_pow = v6.calculate_band_power(after_raw, v6.freq_bands)
        for band in v6.freq_bands:
            before = b_pow.get(band, 0.0)
            after = a_pow.get(band, 0.0)
            change = ((after - before) / before * 100.0) if before else float("nan")
            rows.append({
                "Recording": label,
                "Band": band,
                "Before (µV²)": before,
                "After (µV²)": after,
                "Change %": change,
            })
    return pd.DataFrame(rows)


def _show_dashboard(paired_data: list, suptitle: str) -> None:
    """Render the before/after dashboard plus the comparison table."""
    valid_pairs = [(p, b, a) for p, b, a in paired_data if b is not None and a is not None]
    if not valid_pairs:
        st.error("Need at least one fully-paired recording (Before + After) to render.")
        return

    fig = v6.render_paired_dashboard(
        valid_pairs,
        freq_bands=v6.freq_bands,
        suptitle=suptitle,
    )

    st.subheader("Dashboard")
    st.pyplot(fig, use_container_width=True)

    st.subheader("Band-power comparison (µV²)")
    table = _build_band_power_table(valid_pairs)
    styled = table.style.format({
        "Before (µV²)": _fmt_power,
        "After (µV²)": _fmt_power,
        "Change %": lambda v: f"{v:+.1f}%" if np.isfinite(v) else "—",
    })
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    st.title("🧠 EEG Before/After Dashboard")
    st.markdown(
        "Upload all your `.eeg` files in the single box below and click "
        "**Generate dashboard**. Before/After recordings are paired "
        "automatically by person and mudra."
    )
    st.caption(
        "Naming convention: `{Person}_{state}_{Mudra}.eeg`, where *state* is "
        "`b4`/`before` or `af`/`after` (any casing). e.g. `Rahul_b4_Prana.eeg` "
        "and `Rahul_AF_prana.eeg` form one before/after pair. A single "
        "before file is reused as the baseline for any mudra missing its own."
    )

    cfg = _sidebar_config()

    files = st.file_uploader(
        "EEG files (`{Person}_{state}_{Mudra}.eeg`)",
        type=["eeg"],
        accept_multiple_files=True,
        key="eeg_files",
    )

    if not files:
        st.info("Upload one or more `.eeg` files to enable the dashboard.")

    has_channels = len(cfg["picked"]) > 0
    name_count_ok = len(cfg["ch_names"]) == cfg["num_channels"]
    unique_names = len(set(cfg["ch_names"])) == len(cfg["ch_names"])
    if not has_channels:
        st.warning("Select at least one channel to analyze.")
    if not unique_names:
        st.warning("Channel names must be unique.")
    if not name_count_ok:
        st.warning("Channel name count doesn't match the channel-in-file count.")

    st.divider()

    can_run = (
        bool(files)
        and has_channels
        and unique_names
        and name_count_ok
    )

    if st.button("Generate dashboard", type="primary", disabled=not can_run):
        try:
            paired_data, suptitle = _build_payload(files, cfg)
            _show_dashboard(paired_data, suptitle)
        except Exception as exc:
            st.error(f"Failed to render dashboard: {exc}")
            st.exception(exc)


if __name__ == "__main__":
    main()
