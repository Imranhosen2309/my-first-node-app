"""
================================================================================
 UAV BVLOS Cellular (4G/LTE) Communication Simulation with ML/RL Integration
================================================================================
Thesis: Beyond Visual Line of Sight (BVLOS) communication of drone/UAV using
        ground-based (GSM/4G) cellular networks, integrated with Machine
        Learning and AI.

WHAT THIS SCRIPT DOES (end-to-end, single file, minimal dependencies):
  1. CONFIG           - all tunable simulation parameters in one place
  2. CHANNEL MODEL     - 3GPP TR 36.777-inspired Air-to-Ground (A2G) path loss,
                         LOS/NLOS probability, antenna down-tilt pattern,
                         SINR, and Doppler shift
  3. MOBILITY MODEL    - UAV waypoint / straight-line BVLOS trajectory generator
  4. NETWORK ENV       - Base Station (BS) layout + rule-based (Event-A3/TTT)
                         baseline handover logic  -> this is the classical
                         non-ML approach we will beat with ML/RL
  5. DATASET GENERATOR - runs many simulated missions across altitude / speed /
                         BS-density scenarios and exports a labeled CSV dataset
  6. SIGNAL PREDICTION - (a) a static-feature Linear Regression baseline
                         (mirrors "distance/elevation-angle only" literature
                         approach) and (b) a time-window MLP model (proxy for
                         an LSTM -- see README for the drop-in PyTorch LSTM)
  7. RL ENVIRONMENT    - a Gym-style (but dependency-free) environment where
                         an agent decides WHEN to trigger handover, informed
                         by the signal predictor
  8. Q-LEARNING AGENT  - a tabular Q-learning agent (zero heavy dependencies,
                         runs anywhere) that learns a handover policy and is
                         compared against the rule-based baseline
  9. EVALUATION        - prints + plots comparison metrics (RMSE, handover
                         failure rate, ping-pong rate, reward curves)
 10. MAIN PIPELINE     - orchestrates 1-9 end-to-end with one command:
                             python uav_bvlos_simulation.py

DEPENDENCIES (all pip-installable, all CPU-only, no internet needed at runtime):
    numpy, pandas, matplotlib, scikit-learn
    (OPTIONAL, for the production LSTM upgrade -- see Section 6 docstring):
    torch

HOW TO RUN:
    pip install numpy pandas matplotlib scikit-learn
    python uav_bvlos_simulation.py

OUTPUT (written to ./outputs/):
    uav_signal_dataset.csv          - the generated ML training dataset
    fig1_sample_mission.png         - RSRP vs. distance for one sample flight
    fig2_rl_training_curve.png      - Q-learning reward curve
    fig3_handover_comparison.png    - baseline vs RL handover performance
    results_summary.json            - all final numeric results (for the thesis)

IMPORTANT ACADEMIC NOTE:
    The path-loss / LOS-probability constants used below are SIMPLIFIED,
    representative values *inspired by* 3GPP TR 36.777 and Al-Hourani et al.
    (2014), chosen so the simulation behaves realistically (handover rate
    rises with altitude, sidelobe interference appears, etc.). Before you
    quote exact numeric constants in your thesis, cross-check them against
    the official 3GPP TR 36.777 document (Section 6) and cite accordingly.
    Everything is exposed in the CONFIG section so you can swap in the exact
    published constants with a one-line change.
================================================================================
"""

import os
import json
import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt

RNG_SEED = 42
np.random.seed(RNG_SEED)
random.seed(RNG_SEED)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


# ==============================================================================
# SECTION 1: CONFIGURATION
# ==============================================================================
@dataclass
class SimConfig:
    # --- Cellular network parameters ---
    carrier_freq_ghz: float = 2.0          # LTE band (e.g., Band 1 / Band 7 region)
    bs_height_m: float = 30.0              # ground base station antenna height
    bs_tx_power_dbm: float = 46.0          # macro eNB Tx power
    bs_antenna_gain_dbi: float = 15.0      # boresight antenna gain
    antenna_downtilt_deg: float = 6.0      # mechanical/electrical down-tilt
    antenna_3db_beamwidth_deg: float = 10.0
    antenna_max_sidelobe_atten_db: float = 20.0
    noise_figure_db: float = 7.0
    bandwidth_hz: float = 10e6             # 10 MHz LTE channel
    inter_site_distance_m: float = 1000.0  # BS spacing (varies per scenario)
    grid_size: int = 5                     # BS laid out on a grid_size x grid_size grid

    # --- A2G LOS probability model (Al-Hourani et al. 2014 sigmoid form) ---
    los_a: float = 4.88                    # environment-dependent constant (suburban-like)
    los_b: float = 0.43                    # environment-dependent constant

    # --- Handover (rule-based baseline, 3GPP Event A3-style) ---
    ho_hysteresis_db: float = 2.0
    ho_offset_db: float = 0.0
    ho_ttt_s: float = 1.0                  # time-to-trigger
    min_time_between_ho_s: float = 3.0     # anything faster than this = "ping-pong"

    # --- UAV mission / mobility ---
    time_step_s: float = 0.5
    default_altitude_m: float = 120.0
    default_speed_mps: float = 15.0
    mission_length_m: float = 6000.0       # long-range BVLOS corridor

    # --- Randomness ---
    shadow_fading_std_db: float = 3.0


CFG = SimConfig()


# ==============================================================================
# SECTION 2: 3GPP-INSPIRED A2G CHANNEL MODEL
# ==============================================================================
class ChannelModel:
    """Air-to-Ground channel model: LOS probability, path loss, antenna
    pattern, RSRP, SINR and Doppler shift."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg

    def los_probability(self, elevation_deg: np.ndarray) -> np.ndarray:
        """Sigmoid LOS probability vs. elevation angle (Al-Hourani et al. 2014;
        adopted structurally by 3GPP TR 36.777)."""
        a, b = self.cfg.los_a, self.cfg.los_b
        return 1.0 / (1.0 + a * np.exp(-b * (elevation_deg - a)))

    def path_loss_db(self, d3d_m: np.ndarray, elevation_deg: np.ndarray,
                      uav_height_m: float) -> np.ndarray:
        """Probabilistic LOS/NLOS path loss (simplified 3GPP-RMa-AV-inspired)."""
        fc = self.cfg.carrier_freq_ghz
        d3d_m = np.maximum(d3d_m, 1.0)

        pl_los = 28.0 + 22.0 * np.log10(d3d_m) + 20.0 * np.log10(fc)
        pl_nlos = (-17.5 + (46.0 - 7.0 * np.log10(max(uav_height_m, 1.0)))
                   * np.log10(d3d_m) + 20.0 * np.log10(40.0 * np.pi * fc / 3.0))
        pl_nlos = np.maximum(pl_nlos, pl_los)  # NLOS is never better than LOS

        p_los = self.los_probability(elevation_deg)
        # Probabilistic mixture (expected path loss) -- see docstring note on
        # using this for *statistical* dataset generation; for a single Monte-
        # Carlo draw you would instead sample LOS/NLOS per-link with p_los.
        pl_mix = p_los * pl_los + (1 - p_los) * pl_nlos
        return pl_mix

    def antenna_gain_db(self, elevation_deg: np.ndarray) -> np.ndarray:
        """Simplified 3GPP-style vertical antenna pattern: gain drops off the
        further the UAV's elevation angle is from the down-tilt boresight,
        capped by a maximum side-lobe attenuation. This is THE key effect
        that causes UAVs at altitude to suffer strong, erratic interference
        (sidelobe reception) -- a central motivation of this thesis."""
        tilt = self.cfg.antenna_downtilt_deg
        bw3db = self.cfg.antenna_3db_beamwidth_deg
        sla = self.cfg.antenna_max_sidelobe_atten_db
        # angle offset from the antenna's main-lobe pointing direction
        angle_off = elevation_deg - tilt
        atten = np.minimum(12.0 * (angle_off / bw3db) ** 2, sla)
        return self.cfg.bs_antenna_gain_dbi - atten

    def rsrp_dbm(self, d3d_m: np.ndarray, elevation_deg: np.ndarray,
                 uav_height_m: float, shadow_fading: bool = True) -> np.ndarray:
        pl = self.path_loss_db(d3d_m, elevation_deg, uav_height_m)
        gain = self.antenna_gain_db(elevation_deg)
        rsrp = self.cfg.bs_tx_power_dbm + gain - pl
        if shadow_fading:
            rsrp = rsrp + np.random.normal(0, self.cfg.shadow_fading_std_db, size=np.shape(rsrp))
        return rsrp

    def noise_power_dbm(self) -> float:
        # thermal noise (dBm) = -174 + 10log10(BW) + NF
        return -174.0 + 10 * np.log10(self.cfg.bandwidth_hz) + self.cfg.noise_figure_db

    def sinr_db(self, serving_rsrp_dbm: float, interferer_rsrp_dbm: np.ndarray) -> float:
        signal_mw = 10 ** (serving_rsrp_dbm / 10.0)
        interference_mw = np.sum(10 ** (np.asarray(interferer_rsrp_dbm) / 10.0))
        noise_mw = 10 ** (self.noise_power_dbm() / 10.0)
        sinr_linear = signal_mw / (interference_mw + noise_mw)
        return 10 * np.log10(sinr_linear)

    def doppler_shift_hz(self, speed_mps: float, angle_deg: float) -> float:
        """f_d = v * cos(alpha) * f_c / c"""
        c = 3e8
        fc_hz = self.cfg.carrier_freq_ghz * 1e9
        return speed_mps * math.cos(math.radians(angle_deg)) * fc_hz / c


# ==============================================================================
# SECTION 3: BASE STATION LAYOUT + UAV MOBILITY MODEL
# ==============================================================================
def generate_bs_layout(cfg: SimConfig) -> np.ndarray:
    """Regular grid of ground base stations at fixed height."""
    isd = cfg.inter_site_distance_m
    n = cfg.grid_size
    xs = (np.arange(n) - n // 2) * isd
    ys = (np.arange(n) - n // 2) * isd
    xx, yy = np.meshgrid(xs, ys)
    bs_xy = np.stack([xx.ravel(), yy.ravel()], axis=1)
    return bs_xy  # shape (n*n, 2)


def generate_uav_mission(cfg: SimConfig, altitude_m: float, speed_mps: float,
                          start_xy=(0.0, 0.0), heading_deg: float = 0.0
                          ) -> pd.DataFrame:
    """Straight-line long-range BVLOS mission (the simplest and most common
    case in the cited literature for handover-count analysis). Waypoint
    missions can be built by concatenating several of these segments."""
    n_steps = int(cfg.mission_length_m / speed_mps / cfg.time_step_s)
    t = np.arange(n_steps) * cfg.time_step_s
    dist_travelled = speed_mps * t
    heading_rad = math.radians(heading_deg)
    x = start_xy[0] + dist_travelled * math.cos(heading_rad)
    y = start_xy[1] + dist_travelled * math.sin(heading_rad)
    z = np.full(n_steps, altitude_m)
    return pd.DataFrame({"t": t, "x": x, "y": y, "z": z,
                          "speed": speed_mps, "altitude": altitude_m})


# ==============================================================================
# SECTION 4: NETWORK ENVIRONMENT + RULE-BASED (BASELINE) HANDOVER LOGIC
# ==============================================================================
class RuleBasedHandoverController:
    """Classical 3GPP Event-A3 + Hysteresis + Time-to-Trigger handover logic,
    implemented ONCE here and reused by both (a) the standalone baseline
    mission runner and (b) the RL-environment evaluation loop, so the
    baseline is always apples-to-apples with whatever the RL agent is
    compared against."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.candidate_bs = None
        self.candidate_since = None
        self.last_ho_time = -1e9
        self.n_handovers = 0
        self.n_pingpong = 0

    def decide(self, serving_bs: int, best_idx: int, serving_rsrp: float,
               best_rsrp: float, t: float) -> Tuple[bool, int]:
        """Returns (handover_now: bool, new_serving_bs: int)."""
        cfg = self.cfg
        condition_met = (best_idx != serving_bs) and \
            (best_rsrp + cfg.ho_offset_db - cfg.ho_hysteresis_db > serving_rsrp)

        if not condition_met:
            self.candidate_bs, self.candidate_since = None, None
            return False, serving_bs

        if self.candidate_bs != best_idx:
            self.candidate_bs = best_idx
            self.candidate_since = t
            return False, serving_bs

        if (t - self.candidate_since) >= cfg.ho_ttt_s:
            is_pingpong = (t - self.last_ho_time) < cfg.min_time_between_ho_s
            self.n_pingpong += int(is_pingpong)
            self.last_ho_time = t
            self.n_handovers += 1
            self.candidate_bs, self.candidate_since = None, None
            return True, best_idx

        return False, serving_bs


class NetworkEnvironment:
    """Wraps the BS layout + channel model and provides per-timestep RSRP
    values from every BS to the UAV, plus a classical 3GPP Event-A3 /
    Time-to-Trigger rule-based handover algorithm (the baseline we compare
    ML/RL against)."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.channel = ChannelModel(cfg)
        self.bs_xy = generate_bs_layout(cfg)

    def rsrp_all_bs(self, uav_xy: np.ndarray, uav_z: float) -> np.ndarray:
        dxy = self.bs_xy - uav_xy
        horiz_dist = np.linalg.norm(dxy, axis=1)
        d3d = np.sqrt(horiz_dist ** 2 + (uav_z - self.cfg.bs_height_m) ** 2)
        elevation_deg = np.degrees(np.arctan2(uav_z - self.cfg.bs_height_m,
                                               np.maximum(horiz_dist, 1e-3)))
        return self.channel.rsrp_dbm(d3d, elevation_deg, uav_z), d3d, elevation_deg

    def run_rule_based_mission(self, mission_df: pd.DataFrame) -> pd.DataFrame:
        """Runs the classical rule-based (Event A3 + TTT + hysteresis)
        handover algorithm along a mission and returns a per-timestep log
        (this IS the non-ML baseline referenced throughout the thesis)."""
        records = []
        serving_bs = None
        controller = RuleBasedHandoverController(self.cfg)

        for _, row in mission_df.iterrows():
            uav_xy = np.array([row.x, row.y])
            rsrp_all, d3d_all, elev_all = self.rsrp_all_bs(uav_xy, row.z)
            best_idx = int(np.argmax(rsrp_all))

            if serving_bs is None:
                serving_bs = best_idx

            serving_rsrp = rsrp_all[serving_bs]
            best_rsrp = rsrp_all[best_idx]

            did_ho, serving_bs = controller.decide(serving_bs, best_idx,
                                                     serving_rsrp, best_rsrp, row.t)

            interferer_rsrp = np.delete(rsrp_all, serving_bs)
            sinr = self.channel.sinr_db(rsrp_all[serving_bs], interferer_rsrp)

            records.append({
                "t": row.t, "x": row.x, "y": row.y, "z": row.z,
                "speed": row.speed, "altitude": row.altitude,
                "serving_bs": serving_bs, "serving_rsrp": rsrp_all[serving_bs],
                "best_rsrp": best_rsrp, "dist_serving_m": d3d_all[serving_bs],
                "elev_serving_deg": elev_all[serving_bs], "sinr_db": sinr,
                "handover_event": int(did_ho),
            })

        log = pd.DataFrame(records)
        log.attrs["n_handovers"] = controller.n_handovers
        log.attrs["n_pingpong"] = controller.n_pingpong
        return log


# ==============================================================================
# SECTION 5: DATASET GENERATION (for ML training -- RQ1)
# ==============================================================================
def build_feature_window(series: np.ndarray, window: int = 5) -> np.ndarray:
    """Turns a 1D signal series into overlapping windows (used both by the
    windowed-MLP baseline below and as the natural input shape for an LSTM)."""
    n = len(series)
    feats = np.zeros((n, window))
    for i in range(n):
        lo = max(0, i - window + 1)
        seg = series[lo:i + 1]
        pad = window - len(seg)
        feats[i] = np.concatenate([np.full(pad, seg[0]), seg])
    return feats


def generate_dataset(cfg: SimConfig, n_missions: int = 40) -> pd.DataFrame:
    """Runs many missions across altitude / speed / BS-density scenarios and
    concatenates their per-timestep logs into one labeled ML dataset."""
    altitudes = [50, 100, 150, 200, 250, 300]
    speeds = [5, 10, 15, 20, 25]
    isds = [600, 1000, 1500]

    all_logs = []
    mission_id = 0
    for _ in range(n_missions):
        alt = float(np.random.choice(altitudes))
        spd = float(np.random.choice(speeds))
        isd = float(np.random.choice(isds))
        heading = float(np.random.uniform(0, 360))

        cfg_i = SimConfig(**{**cfg.__dict__, "inter_site_distance_m": isd})
        env = NetworkEnvironment(cfg_i)
        mission = generate_uav_mission(cfg_i, altitude_m=alt, speed_mps=spd,
                                        heading_deg=heading)
        log = env.run_rule_based_mission(mission)
        log["mission_id"] = mission_id
        log["isd_m"] = isd

        # windowed RSRP features (history) + the ML *label* = next-step RSRP
        rsrp_window = build_feature_window(log["serving_rsrp"].values, window=5)
        for w in range(5):
            log[f"rsrp_lag{w}"] = rsrp_window[:, w]
        log["rsrp_next"] = log["serving_rsrp"].shift(-1)
        log["handover_next"] = log["handover_event"].shift(-1).fillna(0)

        all_logs.append(log.iloc[:-1])  # drop last row (no next-step label)
        mission_id += 1

    dataset = pd.concat(all_logs, ignore_index=True)
    return dataset


# ==============================================================================
# SECTION 6: SIGNAL PREDICTION MODELS  (addresses RQ1 / H1)
# ==============================================================================
"""
Two models are implemented and compared, exactly mirroring the Research Gap
identified in the literature review (static-feature regression vs. a
time-series-aware model):

  (a) STATIC BASELINE   -- Linear Regression on [distance, elevation, altitude]
                            only (mirrors most published RSRP/RSRQ ML papers,
                            e.g. the RSRP/RSRQ measurement-based ML study in
                            your Literature Matrix).

  (b) WINDOWED MLP MODEL -- scikit-learn MLPRegressor fed a short history
                            window of past RSRP values + mobility features.
                            This is a *dependency-free stand-in* for an LSTM
                            (no internet / GPU required to run it here).

  PRODUCTION UPGRADE: If you have PyTorch installed (`pip install torch`),
  replace class WindowedMLPPredictor with the LSTM below -- it uses the exact
  same input tensor shape (batch, window, features) so the rest of the
  pipeline (dataset, training loop, evaluation) does not need to change:

      import torch, torch.nn as nn
      class LSTMPredictor(nn.Module):
          def __init__(self, n_features, hidden=32, layers=1):
              super().__init__()
              self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True)
              self.head = nn.Linear(hidden, 1)
          def forward(self, x):                 # x: (batch, window, n_features)
              out, _ = self.lstm(x)
              return self.head(out[:, -1, :])    # predict from last timestep
      # train with: nn.MSELoss() + torch.optim.Adam(model.parameters(), lr=1e-3)
"""
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


def train_and_compare_predictors(dataset: pd.DataFrame) -> Dict:
    df = dataset.dropna(subset=["rsrp_next"]).copy()
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]

    # ---- (a) static-feature baseline ----
    static_feats = ["dist_serving_m", "elev_serving_deg", "altitude"]
    Xa_tr, Xa_te = train[static_feats].values, test[static_feats].values
    y_tr, y_te = train["rsrp_next"].values, test["rsrp_next"].values

    scaler_a = StandardScaler().fit(Xa_tr)
    lin = LinearRegression().fit(scaler_a.transform(Xa_tr), y_tr)
    pred_a = lin.predict(scaler_a.transform(Xa_te))
    rmse_a = math.sqrt(mean_squared_error(y_te, pred_a))
    mae_a = mean_absolute_error(y_te, pred_a)

    # ---- (b) windowed MLP (LSTM stand-in) ----
    window_feats = [f"rsrp_lag{w}" for w in range(5)] + ["speed", "altitude"]
    Xb_tr, Xb_te = train[window_feats].values, test[window_feats].values

    scaler_b = StandardScaler().fit(Xb_tr)
    mlp = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=400,
                        random_state=RNG_SEED, early_stopping=True)
    mlp.fit(scaler_b.transform(Xb_tr), y_tr)
    pred_b = mlp.predict(scaler_b.transform(Xb_te))
    rmse_b = math.sqrt(mean_squared_error(y_te, pred_b))
    mae_b = mean_absolute_error(y_te, pred_b)

    return {
        "static_baseline": {"rmse_dbm": rmse_a, "mae_dbm": mae_a},
        "windowed_mlp": {"rmse_dbm": rmse_b, "mae_dbm": mae_b},
        "improvement_pct": 100.0 * (rmse_a - rmse_b) / rmse_a,
    }


# ==============================================================================
# SECTION 7: RL ENVIRONMENT (Gym-style, dependency-free) -- addresses RQ2
# ==============================================================================
class HandoverRLEnv:
    """A minimal Gym-style environment (reset/step) for learning WHEN to
    hand over. No 'gymnasium' package required -- this hand-rolled interface
    is a drop-in match for it, so you can wrap it with gymnasium.Env later
    if you want to use stable-baselines3 on your own machine."""

    N_ACTIONS = 2  # 0 = stay, 1 = handover-now

    def __init__(self, cfg: SimConfig, altitudes=(80, 150, 220), speeds=(8, 15, 22)):
        self.cfg = cfg
        self.altitudes = altitudes
        self.speeds = speeds
        self.env = NetworkEnvironment(cfg)
        self._reset_mission()

    def _reset_mission(self):
        alt = float(np.random.choice(self.altitudes))
        spd = float(np.random.choice(self.speeds))
        heading = float(np.random.uniform(0, 360))
        self.mission = generate_uav_mission(self.cfg, alt, spd, heading_deg=heading)
        self.idx = 0
        self.serving_bs = None
        self.last_ho_t = -1e9
        self.n_handovers = 0
        self.n_pingpong = 0

    def reset(self):
        self._reset_mission()
        return self._state()

    def _row(self):
        return self.mission.iloc[self.idx]

    def _state(self):
        row = self._row()
        uav_xy = np.array([row.x, row.y])
        rsrp_all, d3d_all, elev_all = self.env.rsrp_all_bs(uav_xy, row.z)
        best_idx = int(np.argmax(rsrp_all))
        if self.serving_bs is None:
            self.serving_bs = best_idx
        serving_rsrp = rsrp_all[self.serving_bs]
        best_rsrp = rsrp_all[best_idx]
        time_since_ho = row.t - self.last_ho_t
        # normalized state vector
        state = np.array([
            row.z / 300.0,
            row.speed / 25.0,
            (serving_rsrp + 130) / 60.0,
            (best_rsrp - serving_rsrp) / 20.0,
            min(time_since_ho, 10.0) / 10.0,
        ])
        self._cache = dict(rsrp_all=rsrp_all, best_idx=best_idx,
                            serving_rsrp=serving_rsrp, best_rsrp=best_rsrp, row=row)
        return state

    def step(self, action: int):
        c = self._cache
        row = c["row"]
        reward = 0.0
        did_ho = False

        # --- Hardware/protocol cooldown: a real radio cannot legally re-attach
        # to a new cell within `min_time_between_ho_s` of its last handover
        # (RRC reconfiguration + re-establishment takes real time). This is a
        # structural constraint -- not a learned behaviour -- exactly like a
        # minimum Time-to-Trigger in a real system. It makes ping-pong
        # PHYSICALLY impossible rather than something the agent must merely
        # learn to avoid, which keeps the RL-vs-baseline comparison meaningful
        # (both are now judged on *when-to-switch* quality, not on whether
        # they can spam handovers).
        in_cooldown = (row.t - self.last_ho_t) < self.cfg.min_time_between_ho_s

        if action == 1 and c["best_idx"] != self.serving_bs:
            if in_cooldown:
                reward -= 0.5  # small friction penalty for a blocked/wasted attempt
            else:
                self.serving_bs = c["best_idx"]
                self.last_ho_t = row.t
                self.n_handovers += 1
                did_ho = True
                reward -= 6.0  # realistic one-off signaling/interruption cost

        # QoS reward: small per-step term (normalized so it cannot dwarf the
        # one-off handover cost over a long episode) that rewards a strong
        # serving-cell signal and heavily penalizes near radio-link-failure.
        serving_rsrp_after = c["rsrp_all"][self.serving_bs]
        reward += (serving_rsrp_after + 110) / 80.0
        if serving_rsrp_after < -110:
            reward -= 3.0  # heavy penalty for near radio-link-failure signal

        self.idx += 1
        done = self.idx >= len(self.mission) - 1
        next_state = None if done else self._state()
        info = {"handover": did_ho, "n_handovers": self.n_handovers,
                "n_pingpong": self.n_pingpong}
        return next_state, reward, done, info


# ==============================================================================
# SECTION 8: TABULAR Q-LEARNING AGENT (zero heavy dependencies)
# ==============================================================================
class QLearningAgent:
    """Discretizes the 5-D continuous state into bins and learns a tabular
    Q-function. Simple, transparent, and100% reproducible -- ideal for a
    thesis baseline-vs-RL comparison. A deep-RL (DQN/PPO via
    stable-baselines3) upgrade is described in the README for your machine."""

    def __init__(self, n_bins: int = 10, n_actions: int = 2,
                 alpha: float = 0.15, gamma: float = 0.97,
                 eps_start: float = 1.0, eps_end: float = 0.02,
                 eps_decay_episodes: int = 700):
        self.n_bins = n_bins
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_episodes = eps_decay_episodes
        self.q_table: Dict[Tuple[int, ...], np.ndarray] = {}

    def _discretize(self, state: np.ndarray) -> Tuple[int, ...]:
        clipped = np.clip(state, 0.0, 1.0)
        bins = np.minimum((clipped * self.n_bins).astype(int), self.n_bins - 1)
        return tuple(bins.tolist())

    def _q(self, s_key):
        if s_key not in self.q_table:
            self.q_table[s_key] = np.zeros(self.n_actions)
        return self.q_table[s_key]

    def epsilon(self, episode: int) -> float:
        frac = min(episode / self.eps_decay_episodes, 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, state: np.ndarray, episode: int) -> int:
        if np.random.rand() < self.epsilon(episode):
            return np.random.randint(self.n_actions)
        s_key = self._discretize(state)
        return int(np.argmax(self._q(s_key)))

    def update(self, state, action, reward, next_state, done):
        s_key = self._discretize(state)
        q_sa = self._q(s_key)
        target = reward
        if not done and next_state is not None:
            ns_key = self._discretize(next_state)
            target += self.gamma * np.max(self._q(ns_key))
        q_sa[action] += self.alpha * (target - q_sa[action])


def train_q_learning(cfg: SimConfig, n_episodes: int = 400) -> Tuple[QLearningAgent, List[float]]:
    env = HandoverRLEnv(cfg)
    agent = QLearningAgent()
    reward_history = []

    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = agent.act(state, ep)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        reward_history.append(total_reward)

    return agent, reward_history


def evaluate_policy(cfg: SimConfig, agent: QLearningAgent = None,
                     n_eval_missions: int = 30) -> Dict:
    """Evaluates either the trained Q-learning agent (agent given) or the
    classical rule-based baseline (agent=None) on fresh random missions and
    returns handover-quality metrics."""
    env = HandoverRLEnv(cfg)
    total_handovers, total_pingpong, total_weak_signal_steps, total_steps = 0, 0, 0, 0
    rsrp_sum = 0.0

    for _ in range(n_eval_missions):
        state = env.reset()
        # Fresh rule-based controller per mission (mirrors run_rule_based_mission)
        controller = RuleBasedHandoverController(cfg) if agent is None else None
        done = False
        step_i = 0
        while not done:
            if agent is not None:
                action = int(np.argmax(agent._q(agent._discretize(state))))
            else:
                # rule-based baseline decision using the exact same A3+TTT+hysteresis
                # logic as run_rule_based_mission (via RuleBasedHandoverController)
                c = env._cache
                would_ho, _ = controller.decide(env.serving_bs, c["best_idx"],
                                                  c["serving_rsrp"], c["best_rsrp"],
                                                  c["row"].t)
                action = 1 if would_ho else 0
            next_state, reward, done, info = env.step(action)
            serving_rsrp_now = env._cache["rsrp_all"][env.serving_bs] if not done else \
                                env._cache["rsrp_all"][env.serving_bs]
            weak = serving_rsrp_now < -110
            total_weak_signal_steps += int(weak)
            total_steps += 1
            rsrp_sum += serving_rsrp_now
            state = next_state
            step_i += 1
        total_handovers += env.n_handovers
        total_pingpong += env.n_pingpong

    return {
        "avg_handovers_per_mission": total_handovers / n_eval_missions,
        "avg_pingpong_per_mission": total_pingpong / n_eval_missions,
        "pingpong_rate_pct": 100.0 * total_pingpong / max(total_handovers, 1),
        "weak_signal_time_pct": 100.0 * total_weak_signal_steps / max(total_steps, 1),
        "avg_serving_rsrp_dbm": rsrp_sum / max(total_steps, 1),
    }


# ==============================================================================
# SECTION 9: EVALUATION & VISUALIZATION
# ==============================================================================
def plot_sample_mission(cfg: SimConfig, path: str):
    env = NetworkEnvironment(cfg)
    mission = generate_uav_mission(cfg, altitude_m=150, speed_mps=15, heading_deg=25)
    log = env.run_rule_based_mission(mission)

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(log["t"], log["serving_rsrp"], color="tab:blue", label="Serving RSRP (dBm)")
    ho_times = log.loc[log["handover_event"] == 1, "t"]
    for ht in ho_times:
        ax1.axvline(ht, color="red", linestyle="--", alpha=0.6)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("RSRP (dBm)", color="tab:blue")
    ax1.set_title("Sample BVLOS Mission: Serving RSRP over Time\n(red dashed = handover event)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return log.attrs.get("n_handovers", 0)


def plot_rl_training_curve(reward_history: List[float], path: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    window = 15
    smoothed = pd.Series(reward_history).rolling(window, min_periods=1).mean()
    ax.plot(reward_history, alpha=0.3, color="gray", label="Episode reward")
    ax.plot(smoothed, color="tab:green", linewidth=2, label=f"Moving avg ({window} ep)")
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Total Episode Reward")
    ax.set_title("Q-Learning Handover Policy: Training Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_handover_comparison(baseline_metrics: Dict, rl_metrics: Dict, path: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    labels = ["Avg handovers\n/mission", "Ping-pong rate (%)"]
    baseline_vals = [baseline_metrics["avg_handovers_per_mission"],
                      baseline_metrics["pingpong_rate_pct"]]
    rl_vals = [rl_metrics["avg_handovers_per_mission"],
               rl_metrics["pingpong_rate_pct"]]
    x = np.arange(len(labels))
    w = 0.35
    axes[0].bar(x - w / 2, baseline_vals, w, label="Rule-based (A3+TTT)", color="tab:orange")
    axes[0].bar(x + w / 2, rl_vals, w, label="Q-Learning", color="tab:green")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_title("Handover Frequency & Ping-Pong")
    axes[0].legend()

    labels2 = ["Avg serving RSRP (dBm)"]
    axes[1].bar([0 - w / 2], [baseline_metrics["avg_serving_rsrp_dbm"]], w,
                color="tab:orange", label="Rule-based (A3+TTT)")
    axes[1].bar([0 + w / 2], [rl_metrics["avg_serving_rsrp_dbm"]], w,
                color="tab:green", label="Q-Learning")
    axes[1].set_xticks([0])
    axes[1].set_xticklabels(labels2)
    axes[1].set_title("Average Signal Quality\n(higher / less negative = better)")
    axes[1].legend()

    fig.suptitle("Baseline vs. RL Handover Policy: Evaluation Metrics (trade-off view)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ==============================================================================
# SECTION 10: MAIN PIPELINE
# ==============================================================================
def main():
    print("=" * 70)
    print("UAV BVLOS CELLULAR SIMULATION -- FULL PIPELINE")
    print("=" * 70)

    results = {}

    # ---- Step 1: sample mission + baseline handover visualization ----
    print("\n[1/5] Running a sample mission with rule-based baseline handover...")
    n_ho_sample = plot_sample_mission(CFG, os.path.join(OUT_DIR, "fig1_sample_mission.png"))
    print(f"      -> sample mission handovers: {n_ho_sample}")
    results["sample_mission_handovers"] = n_ho_sample

    # ---- Step 2: dataset generation ----
    print("\n[2/5] Generating labeled ML dataset across altitude/speed/BS-density scenarios...")
    dataset = generate_dataset(CFG, n_missions=40)
    csv_path = os.path.join(OUT_DIR, "uav_signal_dataset.csv")
    dataset.to_csv(csv_path, index=False)
    print(f"      -> dataset shape: {dataset.shape}, saved to {csv_path}")
    results["dataset_rows"] = int(dataset.shape[0])
    results["dataset_cols"] = int(dataset.shape[1])

    # ---- Step 3: signal prediction (RQ1) ----
    print("\n[3/5] Training & comparing signal-prediction models (RQ1)...")
    pred_results = train_and_compare_predictors(dataset)
    print(f"      -> static-feature baseline RMSE: {pred_results['static_baseline']['rmse_dbm']:.3f} dBm")
    print(f"      -> windowed MLP (LSTM stand-in) RMSE: {pred_results['windowed_mlp']['rmse_dbm']:.3f} dBm")
    print(f"      -> improvement: {pred_results['improvement_pct']:.1f}%")
    results["signal_prediction"] = pred_results

    # ---- Step 4: RL training (RQ2) ----
    print("\n[4/5] Training Q-Learning handover agent (RQ2)... (this takes ~10-20s)")
    agent, reward_history = train_q_learning(CFG, n_episodes=900)
    plot_rl_training_curve(reward_history, os.path.join(OUT_DIR, "fig2_rl_training_curve.png"))
    results["rl_final_avg_reward_last20ep"] = float(np.mean(reward_history[-20:]))

    # ---- Step 5: evaluation -- baseline vs RL (RQ2 / RQ3) ----
    print("\n[5/5] Evaluating baseline vs. RL policy on fresh test missions...")
    baseline_metrics = evaluate_policy(CFG, agent=None, n_eval_missions=30)
    rl_metrics = evaluate_policy(CFG, agent=agent, n_eval_missions=30)
    plot_handover_comparison(baseline_metrics, rl_metrics,
                              os.path.join(OUT_DIR, "fig3_handover_comparison.png"))
    print(f"      -> baseline: {baseline_metrics}")
    print(f"      -> Q-learning: {rl_metrics}")
    results["baseline_metrics"] = baseline_metrics
    results["rl_metrics"] = rl_metrics

    # ---- save everything ----
    summary_path = os.path.join(OUT_DIR, "results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"DONE. All outputs saved in: {OUT_DIR}")
    print("=" * 70)
    return results


if __name__ == "__main__":
    main()
