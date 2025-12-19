import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("NFL WR YPRR Matchup Model")

# -----------------------------
# Helper Functions
# -----------------------------

def route_share_penalty(route_pct, max_penalty=0.80):
    """
    No penalty >= 50%
    Exponential penalty below 50%, maxing at 80% near 5%
    """
    if route_pct >= 0.50:
        return 0.0

    floor = 0.05
    adj = max(route_pct, floor)
    scale = (0.50 - adj) / (0.50 - floor)

    return max_penalty * (scale ** 2)


def compute_adjusted_yprr(row, def_row):
    """
    Coverage-weighted average YPRR
    All defensive inputs assumed as percentages (0–100)
    """
    zone = def_row["zone_rate"] / 100
    man = def_row["man_rate"] / 100

    high0 = def_row["zero_high_rate"] / 100
    high1 = def_row["one_high_rate"] / 100
    high2 = def_row["two_high_rate"] / 100

    coverage_yprr = (
        zone * row["yprr_zone"] +
        man * row["yprr_man"]
    )

    safety_yprr = (
        high0 * row["yprr_0high"] +
        high1 * row["yprr_1high"] +
        high2 * row["yprr_2high"]
    )

    return (coverage_yprr + safety_yprr) / 2


# -----------------------------
# File Uploads
# -----------------------------

wr_file = st.file_uploader("Upload WR Data", type="csv")
def_file = st.file_uploader("Upload Defense Data", type="csv")

if wr_file and def_file:

    wr_df = pd.read_csv(wr_file)
    def_df = pd.read_csv(def_file).set_index("team")

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

        adjusted_yprr = compute_adjusted_yprr(row, defense)

        raw_edge = ((adjusted_yprr / row["base_yprr"]) - 1) * 100
        raw_edge = max(min(raw_edge, 100), -100)

        route_pct = row["routes"] / league_lead_routes
        penalty = route_share_penalty(route_pct)
        final_edge = raw_edge * (1 - penalty)

        results.append({
            "Player": row["player"],
            "Team": row["team"],
            "Opponent": row["opponent"],
            "Route Share": round(route_pct * 100, 1),
            "Base YPRR": round(row["base_yprr"], 2),
            "Adjusted YPRR": round(adjusted_yprr, 2),
            "Edge": round(final_edge, 1)
        })

    results = pd.DataFrame(results)

    # -----------------------------
    # Rank + Ordering
    # -----------------------------

    results = results.sort_values("Edge", ascending=False).reset_index(drop=True)
    results.insert(0, "Rank", results.index + 1)

    # -----------------------------
    # Qualified Toggle
    # -----------------------------

    qualified_only = st.toggle("Qualified (≥35% Route Share)", value=False)

    if qualified_only:
        results = results[results["Route Share"] >= 35]

    # -----------------------------
    # Display Columns (Layout Order)
    # -----------------------------

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
        "📌 **Targets**: Edge ≥ +15, Route Share ≥ 50%\n\n"
        "📌 **Fades**: Edge ≤ −15, Route Share ≥ 50%"
    )

    targets = results[
        (results["Edge"] >= 15) &
        (results["Route Share"] >= 50)
    ]

    fades = results[
        (results["Edge"] <= -15) &
        (results["Route Share"] >= 50)
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
    st.info("Please upload both WR and Defense CSV files to begin.")

