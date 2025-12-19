import streamlit as st
import pandas as pd
import numpy as np
import os

# ------------------------
# Page Setup
# ------------------------
st.set_page_config(page_title="NFL WR Matchup Model", layout="wide")
st.title("NFL WR Coverage Matchup Model (Current Season Only)")

# ------------------------
# Default Data Paths
# ------------------------
BASE_DIR = os.path.dirname(__file__)  # Folder where app.py is located
DEFAULT_WR_PATH = os.path.join(BASE_DIR, "data", "standard_wr_data.csv")
DEFAULT_DEF_PATH = os.path.join(BASE_DIR, "data", "standard_def_data.csv")
DEFAULT_MATCHUP_PATH = os.path.join(BASE_DIR, "data", "standard_matchup_data.csv")
DEFAULT_BLITZ_PATH = os.path.join(BASE_DIR, "data", "standard_blitz_data.csv")

# ------------------------
# Upload Data
# ------------------------
st.sidebar.header("Upload Data Files (Optional)")
wr_file = st.sidebar.file_uploader("WR Data CSV", type="csv")
def_file = st.sidebar.file_uploader("Defense Tendencies CSV", type="csv")
matchup_file = st.sidebar.file_uploader("Weekly Matchups CSV", type="csv")
blitz_file = st.sidebar.file_uploader("Blitz Data CSV", type="csv")

qualified_toggle = st.sidebar.checkbox(
    "Show only qualified players (≥35% league-lead routes)"
)

# ------------------------
# Helper: Load CSV with fallback
# ------------------------
def load_csv(uploaded_file, default_path, name):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    elif os.path.exists(default_path):
        return pd.read_csv(default_path)
    else:
        st.error(f"Error: {name} file not found. Expected at: {default_path}")
        st.stop()

# ------------------------
# Core Model
# ------------------------
def compute_model(
    wr_df,
    def_df,
    max_penalty=0.8,
    exponent=2,
    start_penalty=0.50,
    end_penalty=0.05
):

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

        # ---- Player ratios ----
        man_ratio = row["yprr_man"] / base
        zone_ratio = row["yprr_zone"] / base
        onehigh_ratio = row["yprr_1high"] / base
        twohigh_ratio = row["yprr_2high"] / base
        zerohigh_ratio = row["yprr_0high"] / base
        blitz_ratio = row["yprr_blitz"] / base

        # ---- Coverage weighting ----
        coverage_component = (
            defense["man_pct"] * man_ratio +
            defense["zone_pct"] * zone_ratio
        )

        safety_component = (
            defense["onehigh_pct"] * onehigh_ratio +
            defense["twohigh_pct"] * twohigh_ratio +
            defense["zerohigh_pct"] * zerohigh_ratio
        )

        total_safety = (
            defense["onehigh_pct"] +
            defense["twohigh_pct"] +
            defense["zerohigh_pct"]
        )

        if total_safety > 0:
            safety_component /= total_safety

        blitz_component = defense["blitz_pct"] * blitz_ratio if "blitz_pct" in defense else 0

        expected_ratio = (coverage_component + safety_component + blitz_component) / 3
        adjusted_yprr = base * expected_ratio

        # ---- Edge calculation ----
        raw_edge = (adjusted_yprr - base) / base
        raw_edge = np.clip(raw_edge, -0.25, 0.25)
        edge_score = (raw_edge / 0.25) * 100

        # ---- Route-share penalty (edge only) ----
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
            "player": row["player"],
            "team": row["team"],
            "opponent": opponent,
            "route_share": round(route_share, 1),
            "base_yprr": round(base, 2),
            "adjusted_yprr": round(adjusted_yprr, 2),
            "edge": round(edge_score, 1)
        })

    df = pd.DataFrame(results)
    if df.empty:
        return df

    if qualified_toggle:
        df = df[df["route_share"] >= 0.35]

    # Sort standard table by distance from 0 edge
    df["abs_edge"] = df["edge"].abs()
    df = df.sort_values("abs_edge", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    df = df.drop(columns=["abs_edge"])

    return df

# ------------------------
# Run App
# ------------------------
if True:  # always run
    wr_df = load_csv(wr_file, DEFAULT_WR_PATH, "WR Data")
    def_df_raw = load_csv(def_file, DEFAULT_DEF_PATH, "Defense Data")
    matchup_df = load_csv(matchup_file, DEFAULT_MATCHUP_PATH, "Matchup Data")
    blitz_df = load_csv(blitz_file, DEFAULT_BLITZ_PATH, "Blitz Data")

    # Detect defense index column
    for col in ["team", "defense", "def_team", "abbr"]:
        if col in def_df_raw.columns:
            def_df = def_df_raw.set_index(col)
            break
    else:
        st.error("Defense CSV must include a team column.")
        st.stop()

    # Convert percentages
    for col in ["man_pct", "zone_pct", "onehigh_pct", "twohigh_pct", "zerohigh_pct", "blitz_pct"]:
        if col in def_df.columns:
            def_df[col] = def_df[col] / 100.0
        else:
            def_df[col] = 0.0  # fill missing columns with 0

    # Merge matchups
    wr_df = wr_df.merge(matchup_df, on="team", how="left")
    wr_df = wr_df.merge(blitz_df, on="player", how="left")  # Blitz data per player

    results = compute_model(wr_df, def_df)

    if results.empty:
        st.warning("No players available after filtering.")
        st.stop()

    # ---- Column order ----
    display_cols = [
        "rank",
        "player",
        "team",
        "opponent",
        "route_share",
        "base_yprr",
        "adjusted_yprr",
        "edge"
    ]

    st.subheader("WR Matchup Rankings")
    st.info("Players are sorted by how far their edge is from 0 (largest positive or negative matchups at the top).")
    st.dataframe(results[display_cols])

    # ---- Targets & Fades ----
    min_edge = 7.5
    min_routes = 0.40

    targets = results[
        (results["edge"] >= min_edge) &
        (results["route_share"] >= min_routes)
    ]

    fades = results[
        (results["edge"] <= -min_edge) &
        (results["route_share"] >= min_routes)
    ].sort_values("rank", ascending=True)

    st.subheader("Targets (Best Matchups)")
    st.info(
        "Targets: Edge ≥ +7.5 and ≥ 40% of league-lead routes"
    )
    if not targets.empty:
        st.dataframe(targets[display_cols])
    else:
        st.write("No players meet the target criteria this week.")

    st.subheader("Fades (Worst Matchups)")
    st.info(
        "Fades: Edge ≤ -7.5 and ≥ 40% of league-lead routes"
    )
    if not fades.empty:
        st.dataframe(fades[display_cols])
    else:
        st.write("No players meet the fade criteria this week.")

    # ---- Info Section ----
    st.subheader("Stats Description")
    st.markdown("""
    - **Base YPRR**: Player's Yards per Route Run.
    - **Adjusted YPRR**: YPRR adjusted for opponent's coverage tendencies and blitz.
    - **Edge**: Percentage difference between adjusted YPRR and base YPRR, scaled to show positive and negative matchups.
    - **Route Share**: Player's routes run relative to the league leader.
    """)


