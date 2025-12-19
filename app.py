import streamlit as st
import pandas as pd
import numpy as np

# ------------------------
# Page Setup
# ------------------------
st.set_page_config(page_title="NFL WR Matchup Model", layout="wide")
st.title("NFL WR Coverage Matchup Model (Current Season Only)")

# ------------------------
# Upload Data
# ------------------------
st.sidebar.header("Upload Data Files")
wr_file = st.sidebar.file_uploader("WR Data CSV", type="csv")
def_file = st.sidebar.file_uploader("Defense Tendencies CSV", type="csv")
matchup_file = st.sidebar.file_uploader("Weekly Matchups CSV", type="csv")

# Toggle for qualified players
qualified_toggle = st.sidebar.checkbox("Show only qualified players (≥35% league lead routes)")

# ------------------------
# Core Model
# ------------------------
def compute_model(wr_df, def_df, max_penalty=0.8, exponent=2, start_penalty=0.50, end_penalty=0.05):

    league_lead_routes = wr_df["routes_played"].max()
    results = []

    for _, row in wr_df.iterrows():

        base = row["base_yprr"]
        routes = row["routes_played"]

        # ---- Filters ----
        if base < 0.4 or routes <= 0:
            continue

        opp = row["opponent"]
        if pd.isna(opp) or opp not in def_df.index:
            continue

        defense = def_df.loc[opp]

        # ---- Route share ----
        route_share = routes / league_lead_routes

        # ---- Normalize splits vs baseline ----
        man_ratio = row["yprr_man"] / base
        zone_ratio = row["yprr_zone"] / base
        onehigh_ratio = row["yprr_1high"] / base
        twohigh_ratio = row["yprr_2high"] / base
        zerohigh_ratio = row["yprr_0high"] / base

        # ---- Weighted matchup components ----
        coverage_component = defense["man_pct"] * man_ratio + defense["zone_pct"] * zone_ratio

        # Include 0-high safety in safety_component
        safety_component = (
            defense["onehigh_pct"] * onehigh_ratio +
            defense["twohigh_pct"] * twohigh_ratio +
            defense["zerohigh_pct"] * zerohigh_ratio
        )
        # Normalize by total safety percentage to ensure proper weighting
        total_safety_pct = defense["onehigh_pct"] + defense["twohigh_pct"] + defense["zerohigh_pct"]
        if total_safety_pct > 0:
            safety_component = safety_component / total_safety_pct
        else:
            safety_component = 0

        # ---- Expected ratio (average of coverage + safety) ----
        expected_ratio = (coverage_component + safety_component) / 2

        # ---- Adjusted YPRR (untouched by route share penalty) ----
        adjusted_yprr = base * expected_ratio

        # ---- Raw edge (-25% to +25%) ----
        raw_edge = (adjusted_yprr - base) / base
        raw_edge = np.clip(raw_edge, -0.25, 0.25)
        edge_score = (raw_edge / 0.25) * 100

        # ---- Exponential route share penalty applied only to edge ----
        if route_share >= start_penalty:
            penalty_factor = 0
        elif route_share <= end_penalty:
            penalty_factor = max_penalty
        else:
            penalty_factor = max_penalty * ((start_penalty - route_share) / (start_penalty - end_penalty))**exponent

        edge_score *= (1 - penalty_factor)

        results.append({
            "player": row["player"],
            "team": row["team"],
            "opponent": opp,
            "base_yprr": round(base, 2),
            "adjusted_yprr": round(adjusted_yprr, 2),
            "edge": round(edge_score, 1),
            "route_share": round(route_share, 3)
        })

    df = pd.DataFrame(results)

    if df.empty:
        return df

    # ---- Filter for qualified players if toggle is enabled ----
    if qualified_toggle:
        df = df[df["route_share"] >= 0.35]

    df["rank"] = df["edge"].rank(ascending=False).astype(int)
    return df.sort_values("rank")

# ------------------------
# Run App
# ------------------------
if wr_file and def_file and matchup_file:

    wr_df = pd.read_csv(wr_file)
    def_df_raw = pd.read_csv(def_file)
    matchup_df = pd.read_csv(matchup_file)

    # ---- Detect defense team column ----
    for col in ["team", "defense", "def_team", "abbr"]:
        if col in def_df_raw.columns:
            def_df = def_df_raw.set_index(col)
            break
    else:
        st.error("Defense CSV must contain a team column.")
        st.stop()

    # ---- Convert percentages to decimals automatically ----
    for pct_col in ["man_pct", "zone_pct", "onehigh_pct", "twohigh_pct", "zerohigh_pct"]:
        if pct_col in def_df.columns:
            def_df[pct_col] = def_df[pct_col] / 100.0
        else:
            st.error(f"Defense CSV missing required column: {pct_col}")
            st.stop()

    # ---- Merge weekly matchups ----
    wr_df = wr_df.merge(matchup_df, on="team", how="left")

    # ---- Debug missing defenses ----
    missing_defs = set(wr_df["opponent"].dropna()) - set(def_df.index)
    if missing_defs:
        st.warning(f"Missing defense data for: {', '.join(sorted(missing_defs))}")

    # ---- Run model ----
    results = compute_model(wr_df, def_df)

    if results.empty:
        st.warning("No valid players after filtering.")
        st.stop()

    st.subheader("WR Matchup Rankings")
    st.dataframe(results)

    # ---- Targets/Fades Filters ----
    min_edge = 15  # +15 for targets, -15 for fades
    min_route_share = 0.50  # ≥50% routes for meaningful Targets/Fades

    targets = results[(results["edge"] >= min_edge) & (results["route_share"] >= min_route_share)].sort_values("edge", ascending=False)
    fades = results[(results["edge"] <= -min_edge) & (results["route_share"] >= min_route_share)].sort_values("edge")

    # ---- Targets Section ----
    st.subheader("Targets (Best Matchups)")
    if not targets.empty:
        st.info(f"Showing players with edge ≥ +{min_edge} and route share ≥ {int(min_route_share*100)}% of league lead")
        st.dataframe(targets)
    else:
        st.info(f"No players meet the criteria (edge ≥ +{min_edge} and route share ≥ {int(min_route_share*100)}%)")

    # ---- Fades Section ----
    st.subheader("Fades (Worst Matchups)")
    if not fades.empty:
        st.info(f"Showing players with edge ≤ -{min_edge} and route share ≥ {int(min_route_share*100)}% of league lead")
        st.dataframe(fades)
    else:
        st.info(f"No players meet the criteria (edge ≤ -{min_edge} and route share ≥ {int(min_route_share*100)}%)")

else:
    st.info("Upload WR, Defense, and Matchup CSV files to begin.")


