import streamlit as st
import pandas as pd
import numpy as np

# ------------------------
# Page Setup
# ------------------------
st.set_page_config(page_title="NFL WR Matchup Model", layout="wide")
st.title("NFL WR Coverage + Blitz Matchup Model (Current Season Only)")

# ------------------------
# Upload Data
# ------------------------
st.sidebar.header("Upload Data Files")
wr_file = st.sidebar.file_uploader("WR Data CSV", type="csv")
def_file = st.sidebar.file_uploader("Defense Tendencies CSV", type="csv")
matchup_file = st.sidebar.file_uploader("Weekly Matchups CSV", type="csv")
blitz_file = st.sidebar.file_uploader("WR Blitz YPRR CSV", type="csv")

qualified_toggle = st.sidebar.checkbox(
    "Show only qualified players (≥35% league-lead routes)"
)

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

        # ---- Blitz ratio (mandatory, neutral fallback) ----
        if pd.isna(row["yprr_blitz"]):
            blitz_ratio = 1.0
        else:
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

        coverage_safety_ratio = (coverage_component + safety_component) / 2

        # ---- Blitz weighting ----
        blitz_component = (
            defense["blitz_pct"] * blitz_ratio +
            (1 - defense["blitz_pct"]) * 1.0
        )

        # ---- Final adjusted YPRR ----
        expected_ratio = (coverage_safety_ratio + blitz_component) / 2
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
            "route_share": round(route_share, 3),
            "base_yprr": round(base, 2),
            "adjusted_yprr": round(adjusted_yprr, 2),
            "edge": round(edge_score, 1)
        })

    df = pd.DataFrame(results)
    if df.empty:
        return df

    if qualified_toggle:
        df = df[df["route_share"] >= 0.35]

    df = df.sort_values("edge", ascending=False)
    df["rank"] = range(1, len(df) + 1)

    return df

# ------------------------
# Run App
# ------------------------
if wr_file and def_file and matchup_file and blitz_file:

    wr_df = pd.read_csv(wr_file)
    def_df_raw = pd.read_csv(def_file)
    matchup_df = pd.read_csv(matchup_file)
    blitz_df = pd.read_csv(blitz_file)

    # ---- Normalize names for blitz merge ----
    def normalize_name(name):
        return (
            str(name).lower()
            .replace(".", "")
            .replace(" jr", "")
            .replace(" iii", "")
            .strip()
        )

    wr_df["player_norm"] = wr_df["player"].apply(normalize_name)
    blitz_df["player_norm"] = blitz_df["player"].apply(normalize_name)

    wr_df = wr_df.merge(
        blitz_df[["player_norm", "yprr_blitz"]],
        on="player_norm",
        how="left"
    )

    # Detect defense index column
    for col in ["team", "defense", "def_team", "abbr"]:
        if col in def_df_raw.columns:
            def_df = def_df_raw.set_index(col)
            break
    else:
        st.error("Defense CSV must include a team column.")
        st.stop()

    # Convert percentages
    for col in [
        "man_pct", "zone_pct",
        "onehigh_pct", "twohigh_pct", "zerohigh_pct",
        "blitz_pct"
    ]:
        def_df[col] = def_df[col] / 100.0

    # Merge matchups
    wr_df = wr_df.merge(matchup_df, on="team", how="left")

    results = compute_model(wr_df, def_df)

    if results.empty:
        st.warning("No players available after filtering.")
        st.stop()

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
    st.dataframe(results[display_cols].reset_index(drop=True), hide_index=True)

    # ---- Targets & Fades ----
    min_edge = 15
    min_routes = 0.50

    targets = results[
        (results["edge"] >= min_edge) &
        (results["route_share"] >= min_routes)
    ]

    fades = results[
        (results["edge"] <= -min_edge) &
        (results["route_share"] >= min_routes)
    ]

    st.subheader("Targets (Best Matchups)")
    st.info(
        "Targets must have:\n"
        "• Edge ≥ +15\n"
        "• ≥ 50% of league-lead routes\n"
        "• Adjusted YPRR reflects coverage + safety + blitz"
    )

    if not targets.empty:
        st.dataframe(targets[display_cols].reset_index(drop=True), hide_index=True)
    else:
        st.write("No players meet the target criteria this week.")

    st.subheader("Fades (Worst Matchups)")
    st.info(
        "Fades must have:\n"
        "• Edge ≤ -15\n"
        "• ≥ 50% of league-lead routes\n"
        "• Blitz exposure contributes to downside"
    )

    if not fades.empty:
        st.dataframe(fades[display_cols].reset_index(drop=True), hide_index=True)
    else:
        st.write("No players meet the fade criteria this week.")

else:
    st.info("Upload WR, Defense, Matchup, and Blitz CSV files to begin.")



