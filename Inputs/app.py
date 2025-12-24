import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ------------------------
# Page Setup
# ------------------------
st.set_page_config(page_title="NFL Receiver Matchup Model", layout="wide")
st.markdown("<h1 style='color:#ff6f6f'>Receiver Matchup Weekly Model</h1>", unsafe_allow_html=True)

# ------------------------
# Default Data Paths
# ------------------------
DEFAULT_WR_PATH = "data/standard_wr_data.csv"
DEFAULT_DEF_PATH = "data/standard_def_data.csv"
DEFAULT_MATCHUP_PATH = "data/standard_matchup_data.csv"
DEFAULT_BLITZ_PATH = "data/standard_blitz_data.csv"

# ------------------------
# Upload Data (Optional Overrides)
# ------------------------
st.sidebar.header("Control Panel")

wr_file = st.sidebar.file_uploader("WR Data CSV", type="csv")
def_file = st.sidebar.file_uploader("Defense Tendencies CSV", type="csv")
matchup_file = st.sidebar.file_uploader("Weekly Matchups CSV", type="csv")
blitz_file = st.sidebar.file_uploader("WR Blitz YPRR CSV", type="csv")

# ------------------------
# Route-share filter toggles
# ------------------------
qualified_toggle_35 = st.sidebar.checkbox("Show only players ≥35% route share")
qualified_toggle_20 = st.sidebar.checkbox("Show only players ≥20% route share")

# ------------------------
# Load Data
# ------------------------
try:
    wr_df = pd.read_csv(wr_file) if wr_file else pd.read_csv(DEFAULT_WR_PATH)
    def_df_raw = pd.read_csv(def_file) if def_file else pd.read_csv(DEFAULT_DEF_PATH)
    matchup_df = pd.read_csv(matchup_file) if matchup_file else pd.read_csv(DEFAULT_MATCHUP_PATH)
    blitz_df = pd.read_csv(blitz_file) if blitz_file else pd.read_csv(DEFAULT_BLITZ_PATH)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ------------------------
# Normalize player names
# ------------------------
def normalize_name(name):
    return str(name).lower().replace(".", "").replace(" jr", "").replace(" iii", "").strip()

wr_df["player_norm"] = wr_df["player"].apply(normalize_name)
blitz_df["player_norm"] = blitz_df["player"].apply(normalize_name)

wr_df = wr_df.merge(
    blitz_df[["player_norm", "yprr_blitz"]],
    on="player_norm",
    how="left"
)

# ------------------------
# Prepare Defense Data
# ------------------------
for col in ["team", "defense", "def_team", "abbr"]:
    if col in def_df_raw.columns:
        def_df = def_df_raw.set_index(col)
        break
else:
    st.error("Defense CSV must include a team column.")
    st.stop()

required_cols = [
    "man_pct", "zone_pct",
    "onehigh_pct", "twohigh_pct", "zerohigh_pct",
    "blitz_pct"
]

for col in required_cols:
    if col not in def_df.columns:
        st.error(f"Missing required defense column: {col}")
        st.stop()
    def_df[col] /= 100.0

wr_df = wr_df.merge(matchup_df, on="team", how="left")

# ------------------------
# League Averages
# ------------------------
league_avg_man = def_df["man_pct"].mean()
league_avg_zone = def_df["zone_pct"].mean()
league_avg_1high = def_df["onehigh_pct"].mean()
league_avg_2high = def_df["twohigh_pct"].mean()
league_avg_0high = def_df["zerohigh_pct"].mean()

# ------------------------
# NEW: Regression + Clamping Constants
# ------------------------
REGRESSION_K = 20
MIN_RATIO = 0.6
MAX_RATIO = 1.6

def regress_to_player_base(split_yprr, base_yprr, routes, k=REGRESSION_K):
    if pd.isna(split_yprr) or split_yprr <= 0:
        return base_yprr
    return (split_yprr * routes + base_yprr * k) / (routes + k)

# ------------------------
# Core Model
# ------------------------
def compute_model(
    wr_df,
    def_df,
    max_penalty=0.6,
    exponent=2,
    start_penalty=30,
    end_penalty=5,
    deviation_boost=0.25
):
    results = []

    for _, row in wr_df.iterrows():
        base = row["base_yprr"]
        routes = row["routes_played"]

        if base < 0.4 or routes <= 0:
            continue

        opponent = row["opponent"]
        if pd.isna(opponent) or opponent not in def_df.index:
            continue

        defense = def_df.loc[opponent]

        # ------------------------
        # Player-relative regression (NEW)
        man_yprr = regress_to_player_base(row["yprr_man"], base, routes)
        zone_yprr = regress_to_player_base(row["yprr_zone"], base, routes)
        onehigh_yprr = regress_to_player_base(row["yprr_1high"], base, routes)
        twohigh_yprr = regress_to_player_base(row["yprr_2high"], base, routes)
        zerohigh_yprr = regress_to_player_base(row["yprr_0high"], base, routes)
        blitz_yprr = regress_to_player_base(row.get("yprr_blitz", np.nan), base, routes)

        # Convert to ratios
        man_ratio = man_yprr / base
        zone_ratio = zone_yprr / base
        onehigh_ratio = onehigh_yprr / base
        twohigh_ratio = twohigh_yprr / base
        zerohigh_ratio = zerohigh_yprr / base
        blitz_ratio = blitz_yprr / base

        # ------------------------
        # Ratio clamping (NEW)
        man_ratio = np.clip(man_ratio, MIN_RATIO, MAX_RATIO)
        zone_ratio = np.clip(zone_ratio, MIN_RATIO, MAX_RATIO)
        onehigh_ratio = np.clip(onehigh_ratio, MIN_RATIO, MAX_RATIO)
        twohigh_ratio = np.clip(twohigh_ratio, MIN_RATIO, MAX_RATIO)
        zerohigh_ratio = np.clip(zerohigh_ratio, MIN_RATIO, MAX_RATIO)
        blitz_ratio = np.clip(blitz_ratio, MIN_RATIO, MAX_RATIO)

        man_pct = defense["man_pct"]
        zone_pct = defense["zone_pct"]
        onehigh_pct = defense["onehigh_pct"]
        twohigh_pct = defense["twohigh_pct"]
        zerohigh_pct = defense["zerohigh_pct"]

        # ------------------------
        # System A
        coverage_component = man_pct * man_ratio + zone_pct * zone_ratio
        total_coverage = man_pct + zone_pct

        safety_component = (
            onehigh_pct * onehigh_ratio +
            twohigh_pct * twohigh_ratio +
            zerohigh_pct * zerohigh_ratio
        )
        total_safety = onehigh_pct + twohigh_pct + zerohigh_pct

        if total_safety > 0:
            safety_component /= total_safety

        if total_coverage + total_safety > 0:
            systemA_ratio = (
                coverage_component * total_coverage +
                safety_component * total_safety
            ) / (total_coverage + total_safety)
        else:
            systemA_ratio = (coverage_component + safety_component) / 2

        # ------------------------
        # System B (Deviation)
        coverage_dev = abs(man_pct - league_avg_man) + abs(zone_pct - league_avg_zone)
        safety_dev = (
            abs(onehigh_pct - league_avg_1high) +
            abs(twohigh_pct - league_avg_2high) +
            abs(zerohigh_pct - league_avg_0high)
        )

        if coverage_dev + safety_dev > 0:
            coverage_weight_dev = coverage_dev / (coverage_dev + safety_dev)
            safety_weight_dev = safety_dev / (coverage_dev + safety_dev)
        else:
            coverage_weight_dev = 0.5
            safety_weight_dev = 0.5

        systemB_ratio = (
            coverage_component * coverage_weight_dev +
            safety_component * safety_weight_dev
        )

        # ------------------------
        # Hybrid ratio
        final_ratio = (
            systemA_ratio * (1 - deviation_boost) +
            systemB_ratio * deviation_boost
        )

        # ------------------------
        # Adjusted YPRR & edge (NO EDGE CAP)
        adjusted_yprr = base * ((final_ratio + blitz_ratio) / 2)
        raw_edge = (adjusted_yprr - base) / base
        edge_score = (raw_edge / 0.25) * 100

        # ------------------------
        # Route-share regression
        route_share = row.get("route_share", 0)

        if route_share >= start_penalty:
            penalty = 0
        elif route_share <= end_penalty:
            penalty = max_penalty
        else:
            penalty = max_penalty * (
                (start_penalty - route_share) /
                (start_penalty - end_penalty)
            ) ** exponent

        edge_score *= (1 - penalty)

        results.append({
            "Player": row["player"],
            "Tm": row["team"],
            "Vs.": opponent,
            "Route (%)": route_share,
            "Base YPRR": round(base, 2),
            "Adj. YPRR": round(adjusted_yprr, 2),
            "Matchup (+/-)": round(edge_score * (1 - deviation_boost), 1),
            "Deviation": round(edge_score * deviation_boost, 1),
            "Edge": round(edge_score, 1)
        })

    df = pd.DataFrame(results)
    if df.empty:
        return df

    if qualified_toggle_35:
        df = df[df["Route (%)"] >= 35]
    elif qualified_toggle_20:
        df = df[df["Route (%)"] >= 20]

    df = df.reindex(df["Edge"].abs().sort_values(ascending=False).index)
    df["Rk"] = range(1, len(df) + 1)
    return df

