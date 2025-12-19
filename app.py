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

        # ---- Blitz ratio ----
        blitz_ratio = row.get("yprr_blitz", np.nan)
        if pd.isna(blitz_ratio):
            blitz_ratio = 1.0
        else:
            blitz_ratio = blitz_ratio / base

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
            "Player": row["player"],
            "Team": row["team"],
            "Opponent": opponent,
            "Route Share": route_share,
            "Base YPRR": base,
            "Adjusted YPRR": adjusted_yprr,
            "Edge": edge_score
        })

    df = pd.DataFrame(results)
    if df.empty:
        return df

    if qualified_toggle:
        df = df[df["Route Share"] >= 0.35]

    df = df.sort_values("Edge", ascending=False)
    df["Rank"] = range(1, len(df) + 1)

    return df

# ------------------------
# Color-coding function for Edge
# ------------------------
def color_edge(val):
    if val > 20:
        return "color: darkgreen; font-weight: bold"
    elif 7.5 < val <= 20:
        return "color: green; font-weight: bold"
    elif 0 < val <= 7.5:
        return "color: lightgreen; font-weight: bold"
    elif -7.5 < val <= 0:
        return "color: lightcoral; font-weight: bold"
    elif -20 < val <= -7.5:
        return "color: red; font-weight: bold"
    else:  # val <= -20
        return "color: darkred; font-weight: bold"

# ------------------------
# Run App
# ------------------------
if wr_file and def_file and matchup_file and blitz_file:

    wr_df = pd.read_csv(wr_file)
    def_df_raw = pd.read_csv(def_file)
    matchup_df = pd.read_csv(matchup_file)
    blitz_df = pd.read_csv(blitz_file)

    # ---- Normalize player names for merging ----
    def normalize_name(name):
        return str(name).lower().replace(".", "").replace(" jr", "").replace(" iii", "").strip()

    wr_df["player_norm"] = wr_df["player"].apply(normalize_name)
    blitz_df["player_norm"] = blitz_df["player"].apply(normalize_name)

    # ---- Merge blitz data safely ----
    wr_df = wr_df.merge(
        blitz_df[["player_norm", "yprr_blitz"]],
        on="player_norm",
        how="left"
    )

    # ---- Detect defense index column ----
    for col in ["team", "defense", "def_team", "abbr"]:
        if col in def_df_raw.columns:
            def_df = def_df_raw.set_index(col)
            break
    else:
        st.error("Defense CSV must include a team column.")
        st.stop()

    # ---- Convert percentages ----
    required_cols = [
        "man_pct", "zone_pct",
        "onehigh_pct", "twohigh_pct", "zerohigh_pct",
        "blitz_pct"
    ]
    for col in required_cols:
        if col not in def_df.columns:
            st.error(f"Missing required defense column: {col}")
            st.stop()
        def_df[col] = def_df[col] / 100.0

    # ---- Merge matchups ----
    wr_df = wr_df.merge(matchup_df, on="team", how="left")

    results = compute_model(wr_df, def_df)

    if results.empty:
        st.warning("No players available after filtering.")
        st.stop()

    display_cols = [
        "Rank",
        "Player",
        "Team",
        "Opponent",
        "Route Share",
        "Base YPRR",
        "Adjusted YPRR",
        "Edge"
    ]

    # ---- Formatting ----
    number_format = {
        "Edge": "{:.1f}",
        "Route Share": "{:.1f}",
        "Base YPRR": "{:.2f}",
        "Adjusted YPRR": "{:.2f}"
    }

    st.subheader("WR Matchup Rankings")
    st.dataframe(results[display_cols].style.applymap(color_edge, subset=["Edge"]).format(number_format))

    # ---- Targets & Fades ----
    min_edge = 7.5
    min_routes = 0.40

    targets = results[
        (results["Edge"] >= min_edge) &
        (results["Route Share"] >= min_routes)
    ]

    fades = results[
        (results["Edge"] <= -min_edge) &
        (results["Route Share"] >= min_routes)
    ].sort_values("Edge")  # Sorted ascending for worst fade first

    st.subheader("Targets (Best Matchups)")
    st.info(
        f"Targets must have:\n"
        f"• Edge ≥ +{min_edge}\n"
        f"• ≥ {int(min_routes*100)}% of league-lead routes\n"
        f"• Adjusted YPRR reflects coverage + safety + blitz"
    )
    if not targets.empty:
        st.dataframe(targets[display_cols].style.applymap(color_edge, subset=["Edge"]).format(number_format))
    else:
        st.write("No players meet the target criteria this week.")

    st.subheader("Fades (Worst Matchups)")
    st.info(
        f"Fades must have:\n"
        f"• Edge ≤ -{min_edge}\n"
        f"• ≥ {int(min_routes*100)}% of league-lead routes\n"
        f"• Blitz exposure contributes to downside"
    )
    if not fades.empty:
        st.dataframe(fades[display_cols].style.applymap(color_edge, subset=["Edge"]).format(number_format))
    else:
        st.write("No players meet the fade criteria this week.")

else:
    st.info("Upload WR, Defense, Matchup, and Blitz CSV files to begin.")


