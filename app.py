import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("NFL WR YPRR Matchup Model (Coverage + Safety + Blitz)")

# -----------------------------
# Helper Functions
# -----------------------------

def route_share_penalty(route_pct, max_penalty=0.80):
    """
    No penalty >= 50%
    Exponential penalty below 50%, maxing near 5%
    Affects EDGE ONLY
    """
    if route_pct >= 0.50:
        return 0.0

    floor = 0.05
    adj = max(route_pct, floor)
    scale = (0.50 - adj) / (0.50 - floor)

    return max_penalty * (scale ** 2)


def coverage_safety_factor(row, def_row):
    """
    Weighted average factor based on coverage + safety
    Rates assumed as percentages (0–100)
    """
    zone = def_row["zone_rate"] / 100
    man = def_row["man_rate"] / 100

    high0 = def_row["zero_high_rate"] / 100
    high1 = def_row["one_high_rate"] / 100
    high2 = def_row["two_high_rate"] / 100

    coverage_factor = (
        zone * (row["yprr_zone"] / row["base_yprr"]) +
        man * (row["yprr_man"] / row["base_yprr"])
    )

    safety_factor = (
        high0 * (row["yprr_0high"] / row["base_yprr"]) +
        high1 * (row["yprr_1high"] / row["base_yprr"]) +
        high2 * (row["yprr_2high"] / row["base_yprr"])
    )

    return (coverage_factor + safety_factor) / 2


def blitz_factor(row, def_row):
    """
    Blitz impact behaves like a coverage modifier
    Missing blitz YPRR falls back to neutral (1.0)
    """
    blitz_rate = def_row["blitz_rate"] / 100

    if pd.isna(row["yprr_blitz"]):
        blitz_ratio = 1.0
    else:
        blitz_ratio = row["yprr_blitz"] / row["base_yprr"]

    return (blitz_rate * blitz_ratio) + ((1 - blitz_rate) * 1.0)


# -----------------------------
# File Uploads
# -----------------------------

wr_file = st.file_uploader("Upload WR Data", type="csv")
def_file = st.file_uploader("Upload Defense Data", type="csv")
blitz_file = st.file_uploader("Upload Blitz YPRR Data", type="csv")

if wr_file and def_file and blitz_file:

    wr_df = pd.read_csv(wr_file)
    def_df = pd.read_csv(def_file).set_index("team")
    blitz_df = pd.read_csv(blitz_file)

    # -----------------------------
    # Normalize Player Names
    # -----------------------------

    def normalize_name(name):
        return (
            name.lower()
            .replace(".", "")
            .replace(" jr", "")
            .replace(" iii", "")
            .strip()
        )

    wr_df["player_norm"] = wr_df["player"].apply(normalize_name)
    blitz_df["player_norm"] = blitz_df["player"].apply(normalize_name)

    # Merge blitz YPRR
    wr_df = wr_df.merge(
        blitz_df[["player_norm", "yprr_blitz"]],
        on="player_norm",
        how="left"
    )

    league_lead_routes = wr_df["routes"].max()

    # -----------------------------
    # Model Computation
    # -----------------------------

    results = []

    for _, row in wr_df.iterrows():

        if row["base_yprr"] < 0.4:
            continue

        if row["opponent"] not in def_df.index:
            continue

        defense = def_df.loc[row["opponent"]]

        cov_factor = coverage_safety_factor(row, defense)
        blitz_adj = blitz_factor(row, defense)

        adjusted_yprr = row["base_yprr"] * cov_factor * blitz_adj

        raw_edge = ((adjusted_yprr / row["base_yprr"]) - 1) * 100
        raw_edge = max(min(raw_edge, 100), -100)

        route_pct = row["routes"] / league_lead_routes
        penalty = route_share_penalty(route_pct)

        final_edge = raw_edge * (1 - penalty)

        results.append({
            "Player": row["player"],
            "Team": row["team"],
            "Opponent": row["opponent"],
            "Route Share %": round(route_pct * 100, 1),
            "Base YPRR": round(row["base_yprr"], 2),
            "Adjusted YPRR": round(adjusted_yprr, 2),
            "Edge": round(final_edge, 1)
        })

    results = pd.DataFrame(results)

    # -----------------------------
    # Ranking & Display
    # -----------------------------

    results = results.sort_values("Edge", ascending=False).reset_index(drop=True)
    results.insert(0, "Rank", results.index + 1)

    display_cols = [
        "Rank",
        "Player",
        "Team",
        "Opponent",
        "Route Share %",
        "Base YPRR",
        "Adjusted YPRR",
        "Edge"
    ]

    st.subheader("All Players")
    st.dataframe(
        results[display_cols],
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------
    # Targets & Fades
    # -----------------------------

    st.markdown("---")
    st.subheader("Targets & Fades")

    st.info(
        "📌 Targets: Edge ≥ +15 AND Route Share ≥ 50%\n\n"
        "📌 Fades: Edge ≤ −15 AND Route Share ≥ 50%"
    )

    targets = results[
        (results["Edge"] >= 15) &
        (results["Route Share %"] >= 50)
    ]

    fades = results[
        (results["Edge"] <= -15) &
        (results["Route Share %"] >= 50)
    ]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Targets")
        if targets.empty:
            st.write("No qualifying targets this week.")
        else:
            st.dataframe(
                targets[display_cols],
                use_container_width=True,
                hide_index=True
            )

    with col2:
        st.markdown("### 🚫 Fades")
        if fades.empty:
            st.write("No qualifying fades this week.")
        else:
            st.dataframe(
                fades[display_cols],
                use_container_width=True,
                hide_index=True
            )

else:
    st.info("Please upload WR, Defense, and Blitz CSV files to run the model.")

