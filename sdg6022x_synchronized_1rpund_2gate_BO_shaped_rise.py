# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:35:34 2026

@author: leeau
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:40:17 2026

@author: leeau
"""

# -*- coding: utf-8 -*-
"""
Synchronized two-channel Reset-PUND generator for the Siglent SDG6022X.

Run modes:
  CONTINUOUS  : both channels loop their records forever (no trigger needed).
  EXT_TRIGGER : one edge on rear Aux In/Out plays one record on both channels.

Each synchronized cycle (t = 0 = cycle start):

  CH1: initial 0 V hold, [optional reset pulse, 0 V gap], P, gap, U, gap,
       N, gap, D, 0 V end hold.
       P and U = BO-optimized rising edge (pos_rise_time_s, shape from
       BO_RISE_SHAPE below, scaled to pos_amplitude_v) + flat top (width) +
       linear falling edge (pos_edge_time_s).
       Reset, N and D = linear rising edge + flat top (width) + linear falling edge.
       P/U share amplitude/width/rise/fall; N/D share theirs.
  CH2: programmable offset from CH1 cycle start (or auto-aligned to CH1 P),
       positive pulse P, 0 V gap, positive pulse U, 0 V end hold.

The CH1 and CH2 records have exactly the same number of samples. The shorter
channel block is padded with 0 V to the shared cycle length. The cycle is
repeated cycle_repeats times inside each record.

Requirements:
    pip install numpy matplotlib pyvisa
    NI-VISA or another working VISA backend

Verify the actual voltage at the DUT with an oscilloscope before connecting
a sensitive device.
"""

import math
import time
import numpy as np
import pyvisa
import matplotlib.pyplot as plt


# =============================================================================
# USER CONFIGURATION
# =============================================================================

LIST_RESOURCES_ONLY = False
SEND_TO_INSTRUMENT = True
SHOW_PLOT = True

awg_address = "USB0::0xF4EC::0x1101::SDG6XFCQA00703::INSTR"
awg_timeout_ms = 30000

# Output load: "50" or "INF" (mapped to Siglent "HZ").
# If this does not match the real load, voltages will be off by a factor of 2.
awg_output_load = "50"

CH1 = "C1"
CH2 = "C2"

# -----------------------------
# Common TrueArb settings
# -----------------------------
arb_sample_rate_hz = 500e5     # 10 ns grid: 20 ns edges and 500 ns widths land exactly
max_sample_rate_hz = 3e8       # SDG6000X TrueArb maximum; confirm model/firmware
maximum_arb_points = 20_000_000
minimum_arb_points = 32

# TrueArb interpolation: "HOLD" (closest to the Agilent STEP filter, sharpest
# steps), "LINE" (linear between samples), or None to leave the firmware default.
#arb_interpolation = "HOLD"

tau = 50e-6
rpund_width = 50e-6
gate_pad = 1e-6

# -----------------------------
# CH1: Reset-PUND (every cycle)
# -----------------------------
init_hold_s = 5e-6            # 0 V at the very start of each cycle

# Reset pulse (signed amplitude)
enable_reset_pulse = True
reset_amplitude_v = -0.6       # SIGNED (e.g. -2.0 -> -2.0 V)
reset_width_s = rpund_width        # Flat top (edges NOT included)
reset_edge_time_s = 20e-9      # Rise AND fall, 0-100 % (0 = single-sample step)
gap_after_reset_s = tau     # 0 V between reset and P

# Positive pulses P and U (shared parameters)
pos_amplitude_v = 0.4          # SIGNED, must be > 0
pos_width_s = rpund_width           # Flat top
pos_rise_time_s = 1e-6         # BO-shaped rising edge duration (the shape is stretched to this)
pos_edge_time_s = 20e-9        # Linear FALLING edge only (the rise is pos_rise_time_s)

# BO-optimized rise shape: Bernstein coefficients c_k as fractions of the
# flat-top amplitude, V(s) = pos_amplitude_v * sum_k c_k C(n,k) s^k (1-s)^(n-k),
# n = 5, s = 0..1 over pos_rise_time_s. Only the SHAPE is borrowed; rise time
# and amplitude above are free. Source: Bayesian optimization run 2026-11-01,
# device BTO_PLD219_2, 1.09 us rise, 1.2 V flat top, voltages >= 0 V
# (rise_time_s1.09e-06_rep3): 35.0 % less switching work than a linear rise
# at the same delta_P. The rise starts at 0.48 and ends at 0.76 of the flat
# top, stepping into both (as optimized).
BO_RISE_SHAPE = np.array([0.5709409800860872, 0.0, 0.2821799540406488,
                          0.4649694883726242, 0.0, 0.9060602792740968]) / 1.2

# Negative pulses N and D (shared parameters)
enable_nd_pulses = False        # False -> sequence ends after U (Reset-P-U only)
neg_amplitude_v = -1.0         # SIGNED, must be < 0 (only used if enabled)
neg_width_s = rpund_width           # Flat top
neg_edge_time_s = 20e-9        # Rise AND fall

# Gaps (0 V)
gap_between_pos_s = tau     # P -> U
gap_pos_to_neg_s = tau     # U -> N  (only used if enable_nd_pulses)
gap_between_neg_s = tau    # N -> D  (only used if enable_nd_pulses)

final_hold_s = 50e-6           # 0 V after D, at the end of each cycle

ch1_trigger_delay_s = 0.0      # Once per record (not per cycle)

# -----------------------------
# CH2: two positive pulses per synchronized cycle
# -----------------------------
# If True, CH2 P starts at the same sample as the CH1 P rising edge and
# ch2_offset_from_cycle_start_s is ignored.
align_ch2_p_to_ch1_p = False

pulse1_amplitude_v = 2.5       # Pulse P: 0 V -> this voltage
pulse1_width_s = 10e-6
pulse2_amplitude_v = 2.5       # Pulse U: 0 V -> this voltage
pulse2_width_s = 10e-6
gap_p_to_u_s = tau-pulse1_width_s         # 0 V interval between P and U, need to subtract the gate width 
ch2_end_hold_s = 100e-6        # 0 V hold after U in each cycle
ch2_trigger_delay_s = 0.0      # Keep equal to ch1_trigger_delay_s so CH2 stays
                               # referenced to the CH1 cycle start
                           
ch2_offset_from_cycle_start_s = init_hold_s+reset_edge_time_s+rpund_width+reset_edge_time_s+tau+pos_rise_time_s+rpund_width-pulse1_width_s-gate_pad  # end of P pulse (P rise = pos_rise_time_s)

# -----------------------------
# Repetition and common timing
# -----------------------------
cycle_repeats = 1              # Complete synchronized cycles per record
end_buffer_s = 0.0             # Common 0 V buffer after the full repeated sequence

# External trigger-rate guard (EXT_TRIGGER mode only). Set None to disable.
expected_external_trigger_period_s = 1 / 60
trigger_margin_s = 1e-3

# -----------------------------
# Run mode / trigger / sync settings
# -----------------------------
# "CONTINUOUS"  : both channels loop their records forever, no trigger needed
# "EXT_TRIGGER" : one record per rear-panel Aux In edge
run_mode = "CONTINUOUS"

phase_lock_channels = True     # CONTINUOUS mode: keep CH1/CH2 records aligned

trigger_edge = "RISE"          # Only used in EXT_TRIGGER mode

# Sync output on rear-panel Aux In/Out (CONTINUOUS mode only).
# Disconnect any external trigger source from Aux In/Out before enabling.
enable_ch1_sync = True
sync_type = "CH1"

# Waveform names stored in the AWG
arb_name_ch1 = "PUND_CH1"
arb_name_ch2 = "PU_CH2"

# -----------------------------
# Upload speed / diagnostics
# -----------------------------
RESET_BEFORE_UPLOAD = True     # *RST first (clean state, adds ~reset_settle_s).
                               # False = faster; all settings this script uses are still set explicitly.
reset_settle_s = 2.0           # Pause after *RST (only if RESET_BEFORE_UPLOAD)
upload_settle_s = 0.2          # Pause after each binary waveform upload (was 1 s)
opc_timeout_ms = 5000          # Max wait for *OPC? (was the 30 s VISA timeout)

DO_READBACK = True            # True = print instrument settings after setup (diagnostics)
readback_timeout_ms = 2000     # Short timeout so an unsupported query can't stall 30 s

SHOW_UPLOAD_TIMING = True      # Print elapsed time for each setup step


# =============================================================================
# HELPERS
# =============================================================================

def siglent_load_string(load):
    if str(load).upper() in ("INF", "HZ", "HIGHZ", "HIGH Z"):
        return "HZ"
    return str(load)


def output_limit_v():
    # Conservative output limits used in the original script.
    return 10.0 if siglent_load_string(awg_output_load) == "HZ" else 5.0


def samples_for_duration(duration_s, minimum=0):
    if duration_s < 0:
        raise ValueError("Durations cannot be negative")
    if duration_s == 0:
        return 0
    return max(minimum, int(round(duration_s * arb_sample_rate_hz)))


def zeros(n):
    return np.zeros(int(n), dtype=np.float32)


def make_ramp(v_start, v_end, n):
    """
    n samples moving linearly from v_start to v_end. The last sample equals
    v_end, so the 0-100 % transition takes n sample periods. n = 0 -> no
    samples (a single-sample step between the neighbouring segments).
    """
    if n <= 0:
        return zeros(0)
    k = np.arange(1, n + 1, dtype=np.float64) / n
    return (v_start + (v_end - v_start) * k).astype(np.float32)


def make_bo_rise(amplitude, n):
    """
    n samples of the BO-optimized rise: BO_RISE_SHAPE (Bernstein coefficients,
    fractions of the flat top) evaluated at s = 0..1 inclusive and scaled to
    amplitude -- the same sampling the optimization used. The flat top follows
    directly, with no forced continuity at the end of the rise.
    """
    degree = BO_RISE_SHAPE.size - 1
    s = np.linspace(0.0, 1.0, n)[:, None]
    k = np.arange(degree + 1)
    binomial = np.array([math.comb(degree, i) for i in k], dtype=np.float64)
    shape = (binomial * s ** k * (1.0 - s) ** (degree - k)) @ BO_RISE_SHAPE
    return (amplitude * shape).astype(np.float32)


def assemble(segments):
    """
    Concatenate (label, array, level) segments and return the waveform plus a
    timeline of {label, start_n, len_n, level}. level is None for 0 V holds/gaps.
    """
    timeline, parts, start = [], [], 0
    for label, arr, level in segments:
        timeline.append({"label": label, "start_n": start, "len_n": len(arr), "level": level})
        parts.append(arr)
        start += len(arr)
    return (np.concatenate(parts) if parts else zeros(0)), timeline


def validate_inputs():
    if not (0 < arb_sample_rate_hz <= max_sample_rate_hz):
        raise ValueError(
            f"Sample rate must be between 0 and {max_sample_rate_hz/1e6:.0f} MSa/s"
        )
    if run_mode.upper() not in ("CONTINUOUS", "EXT_TRIGGER"):
        raise ValueError("run_mode must be 'CONTINUOUS' or 'EXT_TRIGGER'")
    if trigger_edge.upper() not in ("RISE", "FALL"):
        raise ValueError("trigger_edge must be 'RISE' or 'FALL'")
    #if arb_interpolation is not None and arb_interpolation.upper() not in ("HOLD", "LINE"):
    #    raise ValueError("arb_interpolation must be 'HOLD', 'LINE', or None")
    if cycle_repeats < 1:
        raise ValueError("cycle_repeats must be at least 1")

    # ---- CH1 P/U ----
    if pos_amplitude_v <= 0:
        raise ValueError("pos_amplitude_v must be > 0 (signed convention)")
    if pos_width_s <= 0:
        raise ValueError("pos_width_s must be > 0")
    if samples_for_duration(pos_rise_time_s) < BO_RISE_SHAPE.size:
        raise ValueError(f"pos_rise_time_s gives fewer than {BO_RISE_SHAPE.size} samples at "
                         f"{arb_sample_rate_hz/1e6:g} MSa/s -- too coarse for the BO rise shape")

    non_negative = {
        "init_hold_s": init_hold_s,
        "final_hold_s": final_hold_s,
        "pos_edge_time_s": pos_edge_time_s,
        "gap_between_pos_s": gap_between_pos_s,
        "ch1_trigger_delay_s": ch1_trigger_delay_s,
        "ch2_trigger_delay_s": ch2_trigger_delay_s,
        "ch2_offset_from_cycle_start_s": ch2_offset_from_cycle_start_s,
        "gap_p_to_u_s": gap_p_to_u_s,
        "ch2_end_hold_s": ch2_end_hold_s,
        "end_buffer_s": end_buffer_s,
    }

    # ---- CH1 N/D (optional) ----
    if enable_nd_pulses:
        if neg_amplitude_v >= 0:
            raise ValueError("neg_amplitude_v must be < 0 (signed convention)")
        if neg_width_s <= 0:
            raise ValueError("neg_width_s must be > 0")
        non_negative["neg_edge_time_s"] = neg_edge_time_s
        non_negative["gap_pos_to_neg_s"] = gap_pos_to_neg_s
        non_negative["gap_between_neg_s"] = gap_between_neg_s

    # ---- CH1 reset (optional) ----
    if enable_reset_pulse:
        if reset_amplitude_v == 0:
            raise ValueError("reset_amplitude_v must be non-zero when the reset is enabled")
        if reset_width_s <= 0:
            raise ValueError("reset_width_s must be > 0")
        if reset_amplitude_v > 0:
            print("NOTE: reset_amplitude_v is positive; the reset pulse will be positive.")
        non_negative["reset_edge_time_s"] = reset_edge_time_s
        non_negative["gap_after_reset_s"] = gap_after_reset_s

    for name, value in non_negative.items():
        if value < 0:
            raise ValueError(f"{name} cannot be negative")

    # ---- CH2 ----
    if pulse1_amplitude_v <= 0 or pulse2_amplitude_v <= 0:
        raise ValueError("Both CH2 pulse amplitudes must be > 0")
    if pulse1_width_s <= 0 or pulse2_width_s <= 0:
        raise ValueError("Both CH2 pulse widths must be > 0")

    # ---- Voltage limits (only levels that are actually used) ----
    limit = output_limit_v()
    ch1_levels = [pos_amplitude_v]
    if enable_nd_pulses:
        ch1_levels.append(neg_amplitude_v)
    if enable_reset_pulse:
        ch1_levels.append(reset_amplitude_v)
    if max(abs(v) for v in ch1_levels) > limit:
        raise ValueError(f"A CH1 pulse level exceeds the conservative +/-{limit:g} V limit")
    if max(pulse1_amplitude_v, pulse2_amplitude_v) > limit:
        raise ValueError(f"A CH2 pulse level exceeds the conservative +/-{limit:g} V limit")


# =============================================================================
# WAVEFORM CONSTRUCTION
# =============================================================================

def build_synchronized_waveforms():
    """Build two equal-length voltage arrays, both starting at record start."""
    validate_inputs()
    fs = arb_sample_rate_hz
    warnings = []

    def n_of(name, duration_s, minimum=0):
        """Samples for a duration, recording a warning if it is not on the grid."""
        n = samples_for_duration(duration_s, minimum)
        if abs(n - duration_s * fs) > 1e-6:
            warnings.append(
                f"{name}: requested {duration_s*1e9:.3f} ns -> actual "
                f"{n/fs*1e9:.3f} ns ({n} samples)"
            )
        return n

    def make_pulse(name, amplitude, width_s, edge_s, bo_rise_s=None):
        """Rising edge + flat top + falling edge (one edge time for both).
        bo_rise_s given: the rising edge is the BO-optimized shape over
        bo_rise_s instead, and edge_s is the falling edge only."""
        edge_n = n_of(f"{name} edge time", edge_s)
        width_n = n_of(f"{name} width", width_s, minimum=1)
        a = float(amplitude)
        if bo_rise_s is None:
            rise_n, rise = edge_n, make_ramp(0.0, a, edge_n)
        else:
            rise_n = n_of(f"{name} BO rise time", bo_rise_s)
            rise = make_bo_rise(a, rise_n)
        wf = np.concatenate((
            rise,
            np.full(width_n, a, dtype=np.float32),
            make_ramp(a, 0.0, edge_n),
        ))
        return wf, rise_n, width_n, edge_n

    # ---- CH1: [Reset] + P + U + [N + D] ----
    pos_wf, pos_rise_n, pos_width_n, pos_fall_n = make_pulse(
        "P/U", pos_amplitude_v, pos_width_s, pos_edge_time_s, bo_rise_s=pos_rise_time_s)
    pulse_shapes = []

    ch1_segments = [("Initial hold", zeros(n_of("Initial hold", init_hold_s)), None)]

    if enable_reset_pulse:
        reset_wf, reset_edge_n, reset_width_n, _ = make_pulse(
            "Reset", reset_amplitude_v, reset_width_s, reset_edge_time_s)
        ch1_segments += [
            ("Reset", reset_wf, reset_amplitude_v),
            ("Gap after reset", zeros(n_of("Gap after reset", gap_after_reset_s)), None),
        ]
        pulse_shapes.append(("Reset", reset_amplitude_v, reset_edge_n, reset_width_n, reset_edge_n))

    ch1_segments += [
        ("P", pos_wf, pos_amplitude_v),
        ("Gap P->U", zeros(n_of("Gap P->U", gap_between_pos_s)), None),
        ("U", pos_wf, pos_amplitude_v),
    ]
    pulse_shapes.append(("P/U", pos_amplitude_v, pos_rise_n, pos_width_n, pos_fall_n))

    if enable_nd_pulses:
        neg_wf, neg_edge_n, neg_width_n, _ = make_pulse(
            "N/D", neg_amplitude_v, neg_width_s, neg_edge_time_s)
        ch1_segments += [
            ("Gap U->N", zeros(n_of("Gap U->N", gap_pos_to_neg_s)), None),
            ("N", neg_wf, neg_amplitude_v),
            ("Gap N->D", zeros(n_of("Gap N->D", gap_between_neg_s)), None),
            ("D", neg_wf, neg_amplitude_v),
        ]
        pulse_shapes.append(("N/D", neg_amplitude_v, neg_edge_n, neg_width_n, neg_edge_n))

    ch1_segments.append(("End hold", zeros(n_of("End hold", final_hold_s)), None))
    ch1_cycle_active, ch1_timeline = assemble(ch1_segments)

    # ---- CH2: offset + P + gap + U + end hold ----
    if align_ch2_p_to_ch1_p:
        ch2_off_n = next(s["start_n"] for s in ch1_timeline if s["label"] == "P")
    else:
        ch2_off_n = n_of("CH2 offset", ch2_offset_from_cycle_start_s)
    p1_n = n_of("CH2 P width", pulse1_width_s, minimum=1)
    p2_n = n_of("CH2 U width", pulse2_width_s, minimum=1)
    ch2_segments = [
        ("Offset", zeros(ch2_off_n), None),
        ("P", np.full(p1_n, pulse1_amplitude_v, dtype=np.float32), pulse1_amplitude_v),
        ("Gap P->U", zeros(n_of("CH2 gap P->U", gap_p_to_u_s)), None),
        ("U", np.full(p2_n, pulse2_amplitude_v, dtype=np.float32), pulse2_amplitude_v),
        ("End hold", zeros(n_of("CH2 end hold", ch2_end_hold_s)), None),
    ]
    ch2_cycle_active, ch2_timeline = assemble(ch2_segments)

    # ---- Shared synchronized cycle: longer block sets the length ----
    cycle_samples = max(len(ch1_cycle_active), len(ch2_cycle_active))
    if len(ch1_cycle_active) == len(ch2_cycle_active):
        cycle_set_by = "both channels (equal)"
    elif len(ch1_cycle_active) > len(ch2_cycle_active):
        cycle_set_by = "CH1"
    else:
        cycle_set_by = "CH2"
    ch1_cycle = np.concatenate((ch1_cycle_active, zeros(cycle_samples - len(ch1_cycle_active))))
    ch2_cycle = np.concatenate((ch2_cycle_active, zeros(cycle_samples - len(ch2_cycle_active))))

    # ---- Trigger delays occur ONCE per record ----
    ch1_delay_n = n_of("CH1 trigger delay", ch1_trigger_delay_s)
    ch2_delay_n = n_of("CH2 trigger delay", ch2_trigger_delay_s)
    ch1_sequence = np.concatenate((zeros(ch1_delay_n), np.tile(ch1_cycle, int(cycle_repeats))))
    ch2_sequence = np.concatenate((zeros(ch2_delay_n), np.tile(ch2_cycle, int(cycle_repeats))))

    common_active_end_n = max(len(ch1_sequence), len(ch2_sequence))
    end_buffer_n = n_of("End buffer", end_buffer_s)
    total_samples = common_active_end_n + end_buffer_n

    ch1 = np.concatenate((ch1_sequence, zeros(total_samples - len(ch1_sequence))))
    ch2 = np.concatenate((ch2_sequence, zeros(total_samples - len(ch2_sequence))))

    if total_samples < minimum_arb_points:
        pad = minimum_arb_points - total_samples
        ch1 = np.concatenate((ch1, zeros(pad)))
        ch2 = np.concatenate((ch2, zeros(pad)))
        total_samples = minimum_arb_points

    if len(ch1) != len(ch2):
        raise RuntimeError("Internal error: CH1 and CH2 waveform lengths do not match")
    if total_samples > maximum_arb_points:
        raise ValueError(
            f"Waveform has {total_samples:,} points per channel; limit is "
            f"{maximum_arb_points:,}. Reduce repeats, duration, or sample rate."
        )

    timing = {
        "ch1_timeline": ch1_timeline,
        "ch2_timeline": ch2_timeline,
        "pulse_shapes": pulse_shapes,
        "ch1_active_n": len(ch1_cycle_active),
        "ch2_active_n": len(ch2_cycle_active),
        "ch2_offset_n": ch2_off_n,
        "cycle_samples": cycle_samples,
        "cycle_set_by": cycle_set_by,
        "ch1_delay_n": ch1_delay_n,
        "ch2_delay_n": ch2_delay_n,
        "end_buffer_n": end_buffer_n,
        "common_active_end_n": common_active_end_n,
        "total_samples": total_samples,
        "total_duration_s": total_samples / fs,
        "warnings": warnings,
    }

    if run_mode.upper() == "EXT_TRIGGER" and expected_external_trigger_period_s is not None:
        available = expected_external_trigger_period_s - trigger_margin_s
        if timing["total_duration_s"] > available:
            raise ValueError(
                f"Record duration ({timing['total_duration_s']*1e3:.6f} ms) exceeds the "
                f"expected trigger period ({expected_external_trigger_period_s*1e3:.6f} ms) "
                f"minus the {trigger_margin_s*1e3:.3f} ms margin."
            )

    return ch1, ch2, timing


# =============================================================================
# INSTRUMENT I/O
# =============================================================================

def amplitude_and_offset(waveform_v):
    """
    AWG amplitude (Vpp) and offset that exactly span the waveform's min/max
    (always including 0 V), so the base lands at exactly 0 V.
    """
    vmin = min(float(np.min(waveform_v)), 0.0)
    vmax = max(float(np.max(waveform_v)), 0.0)
    vpp = vmax - vmin
    if vpp <= 0:
        raise ValueError("Waveform has zero peak-to-peak amplitude")
    return round(vpp, 6), round((vmax + vmin) / 2.0, 6)


def voltage_to_int16_bytes(voltage, vpp, offset):
    """Encode desired volts using the AWG's full signed int16 range."""
    if vpp <= 0:
        raise ValueError("vpp must be > 0")
    codes = np.round(((voltage.astype(np.float64) - offset) / (vpp / 2.0)) * 32767.0)
    codes = np.clip(codes, -32767, 32767).astype("<i2")
    return codes.tobytes()


def open_siglent():
    rm = pyvisa.ResourceManager()
    awg = rm.open_resource(awg_address)
    awg.timeout = awg_timeout_ms
    awg.write_termination = "\n"
    awg.read_termination = "\n"
    awg.chunk_size = 20 * 1024 * 1024
    return rm, awg


def wait_opc(awg, timeout_ms=None):
    """Wait for the AWG to finish pending operations, with a bounded timeout."""
    old_timeout = awg.timeout
    awg.timeout = timeout_ms if timeout_ms is not None else opc_timeout_ms
    try:
        awg.query("*OPC?")
    except pyvisa.errors.VisaIOError:
        # No answer in time: clear the interface so a late reply can't
        # corrupt the next query, then continue.
        try:
            awg.clear()
        except Exception:
            pass
        time.sleep(0.2)
    finally:
        awg.timeout = old_timeout


def list_visa_resources():
    rm = pyvisa.ResourceManager()
    try:
        print("\nVISA resources found:")
        resources = rm.list_resources()
        if not resources:
            print("  (none) - check USB cable and VISA installation")
        for resource in resources:
            try:
                inst = rm.open_resource(resource)
                inst.timeout = 3000
                idn = inst.query("*IDN?").strip()
                inst.close()
            except Exception as exc:
                idn = f"(could not query: {exc})"
            print(f"  {resource}\n      -> {idn}")
    finally:
        rm.close()


def readback(awg, ch):
    """Print instrument settings for one channel (diagnostics only)."""
    queries = (
        f"{ch}:BSWV?",
        f"{ch}:ARWV?",
        f"{ch}:SRATE?",
        f"{ch}:BTWV?",
        f"{ch}:SYNC?",
        f"{ch}:OUTP?",
    )
    old_timeout = awg.timeout
    awg.timeout = readback_timeout_ms
    try:
        for query in queries:
            try:
                print(f"  {query:<12} -> {awg.query(query).strip()}")
            except pyvisa.errors.VisaIOError as exc:
                print(f"  {query:<12} -> (no response: {exc})")
    finally:
        awg.timeout = old_timeout

def upload_arb(awg, ch, name, waveform_v, vpp, offset):
    """Upload one voltage waveform and configure TrueArb at the set sample rate."""
    arb_frequency = arb_sample_rate_hz / len(waveform_v)
    header = (
        f"{ch}:WVDT WVNM,{name},FREQ,{arb_frequency},AMPL,{vpp},"
        f"OFST,{offset},PHASE,0,WAVEDATA,"
    )
    payload = voltage_to_int16_bytes(waveform_v, vpp, offset)
    print(f"Uploading {ch}: {name} ({len(waveform_v):,} points, {len(payload):,} bytes), "
          f"AMP {vpp} Vpp, OFST {offset} V...")
    awg.write_raw(header.encode("ascii") + payload)
    if upload_settle_s > 0:
        time.sleep(upload_settle_s)
    wait_opc(awg, timeout_ms=awg_timeout_ms)   # large records may need longer

    awg.write(f"{ch}:BSWV WVTP,ARB")
    wait_opc(awg)
    awg.write(f"{ch}:ARWV NAME,{name}")
    wait_opc(awg)
    # Plain TrueArb command (the form this firmware accepts; no interpolation).
    awg.write(f"{ch}:SRATE MODE,TARB,VALUE,{int(round(arb_sample_rate_hz))}")
    wait_opc(awg)
    awg.write(f"{ch}:BSWV AMP,{vpp},OFST,{offset}")
    wait_opc(awg)


# =============================================================================
# REPORTING
# =============================================================================

def plot_waveforms(ch1, ch2, timing):
    fs = arb_sample_rate_hz
    n = len(ch1)
    t_us = np.arange(n, dtype=np.float64) / fs * 1e6
    stride = max(1, int(np.ceil(n / 200_000)))  # decimate full views only
    drawstyle = "default"

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
    ax2.sharex(ax1)

    # ---- Full record ----
    ax1.plot(t_us[::stride], ch1[::stride], linewidth=1.0, color="C0")
    ax1.set_ylabel("CH1 (V)")
    reset_txt = f"Reset {reset_amplitude_v:+g} V + " if enable_reset_pulse else ""
    seq_name = "PUND" if enable_nd_pulses else "PU"
    nd_txt = f", N/D {neg_amplitude_v:+g} V" if enable_nd_pulses else ""
    ax1.set_title(f"CH1: {reset_txt}{seq_name} (P/U {pos_amplitude_v:+g} V, "
                  f"BO-shaped {pos_rise_time_s*1e6:g} µs rise{nd_txt}), "
                  f"{cycle_repeats} cycle(s)/record")

    ax2.plot(t_us[::stride], ch2[::stride], linewidth=1.0, color="C1")
    ax2.set_ylabel("CH2 (V)")
    ax2.set_xlabel("Time from record start (µs)")
    ax2.set_title("CH2: P-U pulse pair, referenced to CH1 cycle start")

    for ax in (ax1, ax2):
        ax.axhline(0, color="black", linewidth=0.7, alpha=0.5)
        ax.grid(True, alpha=0.3)
        for k in range(cycle_repeats):
            start = (timing["ch1_delay_n"] + k * timing["cycle_samples"]) / fs * 1e6
            ax.axvline(start, color="gray", linestyle=":", alpha=0.5)

    # ---- Zoom on the first CH1 Reset-PUND block ----
    s0 = timing["ch1_delay_n"]
    margin = max(20, int(0.05 * timing["ch1_active_n"]))
    s1 = min(n, s0 + timing["ch1_active_n"] + margin)
    ax3.plot(t_us[s0:s1], ch1[s0:s1], linewidth=1.8, color="C0", label="CH1",
             drawstyle=drawstyle)
    ax3.plot(t_us[s0:s1], ch2[s0:s1], linewidth=1.2, color="C1", linestyle="--",
             label="CH2", drawstyle=drawstyle)
    ax3.axhline(0, color="black", linewidth=0.7, alpha=0.5)

    levels = [abs(s["level"]) for s in timing["ch1_timeline"] if s["level"] is not None]
    label_off = 0.15 * max(levels) if levels else 0.2
    colors = {"Reset": "red", "P": "green", "U": "green", "N": "darkorange", "D": "darkorange"}
    for s in timing["ch1_timeline"]:
        if s["level"] is None or s["len_n"] == 0:
            continue
        center_us = (s0 + s["start_n"] + s["len_n"] / 2) / fs * 1e6
        y = s["level"] + (label_off if s["level"] > 0 else -label_off)
        c = colors.get(s["label"], "black")
        ax3.axvline(center_us, color=c, linestyle=":", alpha=0.6)
        ax3.text(center_us, y, s["label"], ha="center", va="center", color=c,
                 fontweight="bold", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax3.set_xlabel("Time from record start (µs)")
    ax3.set_ylabel("Voltage (V)")
    ax3.set_title(f"Zoom: first CH1 cycle ({timing['ch1_active_n']/fs*1e6:.3f} µs active, "
                  f"{arb_sample_rate_hz/1e6:g} MSa/s)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="lower right")
    lo = min(np.min(ch1[s0:s1]), np.min(ch2[s0:s1]), 0.0)
    hi = max(np.max(ch1[s0:s1]), np.max(ch2[s0:s1]), 0.0)
    ax3.set_ylim(lo - 2 * label_off, hi + 2 * label_off)

    plt.tight_layout()
    plt.show()


def print_timeline(timeline, fs, title):
    print(title)
    print(f"  {'Start (µs)':>11}  {'Dur (µs)':>10}  Segment")
    for s in timeline:
        if s["len_n"] == 0:
            continue
        level = f"{s['level']:+g} V" if s["level"] is not None else "0 V"
        print(f"  {s['start_n']/fs*1e6:11.3f}  {s['len_n']/fs*1e6:10.3f}  {s['label']} ({level})")


def print_summary(ch1, ch2, timing):
    fs = arb_sample_rate_hz
    us = lambda n_: n_ / fs * 1e6

    print("\n" + "=" * 72)
    print("SYNCHRONIZED SDG6022X RESET-PUND / P-U SEQUENCE")
    print("=" * 72)
    seq = ("Reset-" if enable_reset_pulse else "") + ("P-U-N-D" if enable_nd_pulses else "P-U")
    print(f"CH1 sequence: {seq}")

    print_timeline(timing["ch1_timeline"], fs, "CH1 cycle (t = 0 = cycle start):")
    print(f"  {us(timing['ch1_active_n']):11.3f}  {'':10}  CH1 active block ends")
    print("CH1 pulse shapes (P/U rise = BO shape, other edges = 0-100 % linear ramps, "
          "width = flat top):")
    for name, amp, rise_n, width_n, fall_n in timing["pulse_shapes"]:
        print(f"  {name:<6} {amp:+.3f} V, rise {rise_n/fs*1e9:.1f} ns, flat {width_n/fs*1e9:.1f} ns, "
              f"fall {fall_n/fs*1e9:.1f} ns, total {(rise_n + width_n + fall_n)/fs*1e9:.1f} ns")

    print_timeline(timing["ch2_timeline"], fs, "CH2 cycle (relative to CH1 cycle start):")
    if align_ch2_p_to_ch1_p:
        print("  (CH2 P auto-aligned to the CH1 P rising-edge start)")
    if timing["ch1_delay_n"] != timing["ch2_delay_n"]:
        shift = us(timing["ch2_delay_n"] - timing["ch1_delay_n"])
        print(f"  NOTE: trigger delays differ, so CH2 is shifted by {shift:+.3f} µs "
              f"relative to the CH1 cycle start.")

    if timing["warnings"]:
        print("-" * 72)
        print("SAMPLE-GRID ROUNDING (durations not a multiple of "
              f"{1e9/fs:.3f} ns):")
        for w in timing["warnings"]:
            print(f"  ! {w}")

    print("-" * 72)
    print(f"CH1 active block                 : {us(timing['ch1_active_n']):.3f} µs")
    print(f"CH2 active block                 : {us(timing['ch2_active_n']):.3f} µs")
    print(f"Shared synchronized cycle        : {us(timing['cycle_samples']):.3f} µs "
          f"(set by {timing['cycle_set_by']})")
    print(f"Complete cycles per record       : {cycle_repeats}")
    print(f"CH1 / CH2 trigger delay          : {us(timing['ch1_delay_n']):.3f} / "
          f"{us(timing['ch2_delay_n']):.3f} µs")
    print(f"Final common buffer              : {us(timing['end_buffer_n']):.3f} µs")
    print(f"Total record duration            : {timing['total_duration_s']*1e6:.3f} µs")
    print(f"Samples per channel              : {timing['total_samples']:,}")
    print(f"Sample rate                      : {fs/1e6:.3f} MSa/s (TrueArb)")
    ch1_vpp, ch1_ofs = amplitude_and_offset(ch1)
    ch2_vpp, ch2_ofs = amplitude_and_offset(ch2)
    print(f"CH1 range / AWG setting          : {np.min(ch1):+.4g} to {np.max(ch1):+.4g} V "
          f"(AMP {ch1_vpp} Vpp, OFST {ch1_ofs:+} V)")
    print(f"CH2 range / AWG setting          : {np.min(ch2):+.4g} to {np.max(ch2):+.4g} V "
          f"(AMP {ch2_vpp} Vpp, OFST {ch2_ofs:+} V)")
    print(f"Output load                      : {awg_output_load}")
    if run_mode.upper() == "CONTINUOUS":
        print(f"Run mode                         : CONTINUOUS, phase-locked={phase_lock_channels}, "
              f"CH1 sync={enable_ch1_sync}")
        print(f"Record repetition rate           : {1.0/timing['total_duration_s']:.6g} Hz")
    else:
        print(f"External trigger                 : {trigger_edge.upper()} edge, rear Aux In/Out, "
              f"expected period {expected_external_trigger_period_s*1e3:.3f} ms")
    print(f"CH1 and CH2 equal sample count   : {len(ch1) == len(ch2)}")
    print("=" * 72)


# =============================================================================
# INSTRUMENT SETUP
# =============================================================================

def setup_awg(ch1_waveform_v, ch2_waveform_v, timing):
    """Upload both records and configure continuous or triggered playback."""
    if len(ch1_waveform_v) != len(ch2_waveform_v):
        raise ValueError("CH1 and CH2 must have exactly the same number of samples")

    load = siglent_load_string(awg_output_load)
    ch1_vpp, ch1_offset = amplitude_and_offset(ch1_waveform_v)
    ch2_vpp, ch2_offset = amplitude_and_offset(ch2_waveform_v)

    limit = output_limit_v()
    for name, wf in (("CH1", ch1_waveform_v), ("CH2", ch2_waveform_v)):
        if np.min(wf) < -limit or np.max(wf) > limit:
            raise ValueError(f"{name} waveform exceeds the conservative +/-{limit:g} V limit")

    t0 = time.perf_counter()

    def step(msg):
        if SHOW_UPLOAD_TIMING:
            print(f"  [{time.perf_counter() - t0:6.2f} s] {msg}")

    rm, awg = open_siglent()
    try:
        print("\nConnected to:", awg.query("*IDN?").strip())
        print("Configuring both channels...")

        awg.write(f"{CH1}:OUTP OFF")
        awg.write(f"{CH2}:OUTP OFF")

        if RESET_BEFORE_UPLOAD:
            awg.write("*RST")
            time.sleep(reset_settle_s)
            wait_opc(awg)
            awg.write(f"{CH1}:OUTP OFF")
            awg.write(f"{CH2}:OUTP OFF")
            step("Instrument reset")

        awg.write(f"{CH1}:OUTP LOAD,{load}")
        awg.write(f"{CH2}:OUTP LOAD,{load}")
        wait_opc(awg)
        step("Output load set")

        upload_arb(awg, CH1, arb_name_ch1, ch1_waveform_v, ch1_vpp, ch1_offset)
        step("CH1 uploaded and configured")
        upload_arb(awg, CH2, arb_name_ch2, ch2_waveform_v, ch2_vpp, ch2_offset)
        step("CH2 uploaded and configured")

        if run_mode.upper() == "CONTINUOUS":
            for ch in (CH1, CH2):
                awg.write(f"{ch}:BTWV STATE,OFF")
            wait_opc(awg)
            awg.write("MODE PHASE-LOCKED" if phase_lock_channels else "MODE INDEPENDENT")
            wait_opc(awg)
        else:
            for ch in (CH1, CH2):
                awg.write(f"{ch}:BTWV STATE,ON")
                wait_opc(awg)
                awg.write(f"{ch}:BTWV GATE_NCYC,NCYC")
                awg.write(f"{ch}:BTWV TRSR,EXT")
                awg.write(f"{ch}:BTWV EDGE,{trigger_edge.upper()}")
                awg.write(f"{ch}:BTWV TIME,1")
                awg.write(f"{ch}:BTWV DLAY,0")
                awg.write(f"{ch}:BTWV STPS,0")
                wait_opc(awg)
            awg.write(f"{CH1}:BURSt:RSTAT RUN")
            wait_opc(awg)
            awg.write(f"{CH2}:BURSt:RSTAT RUN")
            wait_opc(awg)
        step(f"{run_mode.upper()} mode configured")

        # Sync output: only in CONTINUOUS mode (Aux In/Out is the trigger input otherwise).
        awg.write(f"{CH2}:SYNC OFF")
        if run_mode.upper() == "CONTINUOUS" and enable_ch1_sync:
            awg.write(f"{CH1}:SYNC ON,TYPE,{sync_type.upper()}")
        else:
            awg.write(f"{CH1}:SYNC OFF")
        wait_opc(awg)

        awg.write(f"{CH1}:OUTP ON")
        awg.write(f"{CH2}:OUTP ON")
        wait_opc(awg)
        step("Sync set, outputs ON")

        if DO_READBACK:
            print("\nRead-back of CH1:")
            readback(awg, CH1)
            print("\nRead-back of CH2:")
            readback(awg, CH2)
            old_timeout = awg.timeout
            awg.timeout = readback_timeout_ms
            try:
                print(f"\n  MODE?        -> {awg.query('MODE?').strip()}")
            except pyvisa.errors.VisaIOError as exc:
                print(f"\n  MODE?        -> (no response: {exc})")
            finally:
                awg.timeout = old_timeout
            step("Read-back done")

        if run_mode.upper() == "CONTINUOUS":
            rep_s = timing["total_duration_s"]
            sync_txt = ", CH1 sync ON" if enable_ch1_sync else ""
            print(f"\nBoth channels running continuously{sync_txt}. The {cycle_repeats}-cycle "
                  f"record repeats every {rep_s*1e6:.3f} µs ({1.0/rep_s:.6g} Hz).")
        else:
            print(f"\nBoth channels armed. One {trigger_edge.lower()} external trigger should "
                  f"start the complete {cycle_repeats}-cycle sequence "
                  f"({timing['total_duration_s']*1e6:.3f} µs) on CH1 and CH2.")
        print(f"Total setup time: {time.perf_counter() - t0:.2f} s")
    except Exception:
        # Never leave a half-configured output on.
        try:
            awg.write(f"{CH1}:OUTP OFF")
            awg.write(f"{CH2}:OUTP OFF")
        except Exception:
            pass
        raise
    finally:
        awg.close()
        rm.close()


def turn_off_awg():
    rm, awg = open_siglent()
    try:
        awg.write(f"{CH1}:OUTP OFF")
        awg.write(f"{CH2}:OUTP OFF")
        wait_opc(awg)
        print("Both AWG outputs turned OFF.")
    finally:
        awg.close()
        rm.close()


def main():
    if LIST_RESOURCES_ONLY:
        list_visa_resources()
        return

    try:
        ch1, ch2, timing = build_synchronized_waveforms()
    except ValueError as exc:
        print(f"\nCONFIGURATION ERROR: {exc}")
        return

    print_summary(ch1, ch2, timing)

    print_summary(ch1, ch2, timing)

    if SEND_TO_INSTRUMENT:
        try:
            setup_awg(ch1, ch2, timing)
        except pyvisa.errors.VisaIOError as exc:
            print(f"\nVISA ERROR: {exc}")
            print("Check the USB address, VISA installation, and USB connection.")
        except ValueError as exc:
            print(f"\nCONFIGURATION ERROR: {exc}")
    else:
        print("\nSEND_TO_INSTRUMENT=False: waveform was built only.")

    # Plot AFTER uploading: plt.show() can block until the window is closed.
    if SHOW_PLOT:
        plot_waveforms(ch1, ch2, timing)


if __name__ == "__main__":
    main()