import streamlit as st
import pandas as pd
import numpy as np

# ------------------------
# Page Setup
# ------------------------
st.set_page_config(page_title="NFL Receiver Matchup Model", layout="wide")
st.title("Receiver Matchup Weekly Model")

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
st.sidebar.header("Optional: Upload Your Own Data")

wr_file = st.sidebar.file_uploader("WR Data CSV", type="csv")
def_file = st.sidebar.file_uploader("Defense Tendencies CSV", type="csv")
matchup_file = st.sidebar.file_uploader("Weekly Matchups CSV", type="csv")
blitz_file = st.sidebar.file_uploader("WR Blitz YPRR CSV", type="csv")

qualified_toggle = st.sidebar.checkbox(
    "Show only qualified players (≥35% league-lead routes)"
)

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
# Normalize Names & Merge Blitz
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
    def_df[col] = def_df[col] / 100.0

wr_df = wr_df.merge(matchup_df, on="team", how="left")

# ------------------------
# Core Model
# ------------------------
def compute_model(wr_df, def_df):

    league_lead_routes = wr_df["routes_played"].max()
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
        route_share = routes / league_lead_routes

        man_ratio = row["yprr_man"] / base
        zone_ratio = row["yprr_zone"] / base
        onehigh_ratio = row["yprr_1high"] / base
        twohigh_ratio = row["yprr_2high"] / base
        zerohigh_ratio = row["yprr_0high"] / base

        blitz_ratio = row["yprr_blitz"]
        blitz_ratio = 1.0 if pd.isna(blitz_ratio) else blitz_ratio / base

        coverage_component = (
            defense["man_pct"] * man_ratio +
            defense["zone_pct"] * zone_ratio
        )

        safety_component = (
            defense["onehigh_pct"] * onehigh_ratio +
            defense["twohigh_pct"] * twohigh_ratio +
            defense["zerohigh_pct"] * zerohigh_ratio
        )

        total_safety = defense["onehigh_pct"] + defense["twohigh_pct"] + defense["zerohigh_pct"]
        safety_component /= total_safety if total_safety > 0 else 1

        coverage_safety_ratio = (coverage_component + safety_component) / 2

        blitz_component = (
            defense["blitz_pct"] * blitz_ratio +
            (1 - defense["blitz_pct"]) * 1.0
        )

        expected_ratio = (coverage_safety_ratio + blitz_component) / 2
        adjusted_yprr = base * expected_ratio

        raw_edge = np.clip((adjusted_yprr - base) / base, -0.25, 0.25)
        edge = (raw_edge / 0.25) * 100

        if route_share < 0.50:
            penalty = ((0.50 - route_share) / 0.45) ** 2 * 0.8
            edge *= (1 - penalty)

        results.append({
            "Player": row["player"],
            "Team": row["team"],
            "Opponent": opponent,
            "Route Share (%)": round(route_share * 100, 1),
            "Base YPRR": round(base, 2),
            "Adjusted YPRR": round(adjusted_yprr, 2),
            "Edge": round(edge, 1)
        })

    df = pd.DataFrame(results)

    if qualified_toggle:
        df = df[df["Route Share (%)"] >= 35]

    df = df.reindex(df["Edge"].abs().sort_values(ascending=False).index)
    df["Rank"] = range(1, len(df) + 1)

    return df

# ------------------------
# Run Model
# ------------------------
results = compute_model(wr_df, def_df)

# ------------------------
# TEAM FILTER (NEW)
# ------------------------
st.sidebar.header("Team Filter")

team_options = sorted(results["Team"].unique())
selected_teams = st.sidebar.multiselect(
    "Type or select teams to display (leave empty for all)",
    team_options
)

if selected_teams:
    results = results[results["Team"].isin(selected_teams)]

# ------------------------
# Display
# ------------------------
display_cols = [
    "Rank", "Player", "Team", "Opponent",
    "Route Share (%)", "Base YPRR", "Adjusted YPRR", "Edge"
]

st.subheader("Player Rankings")
st.dataframe(results[display_cols])

# ------------------------
# Targets & Fades
# ------------------------
targets = results[(results["Edge"] >= 7.5) & (results["Route Share (%)"] >= 40)]
fades = results[(results["Edge"] <= -7.5) & (results["Route Share (%)"] >= 40)].sort_values("Edge")

st.subheader("Targets")
st.dataframe(targets[display_cols])

st.subheader("Fades")
st.dataframe(fades[display_cols])
