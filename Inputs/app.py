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
qualified_toggle_35 = st.sidebar.checkbox(
    "Show only players ≥35% route share"
)
qualified_toggle_20 = st.sidebar.checkbox(
    "Show only players ≥20% route share"
)

# ------------------------
# Load Data (Default or Uploaded)
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
# Normalize player names for merging
# ------------------------
def normalize_name(name):
    return str(name).lower().replace(".", "").replace(" jr", "").replace(" iii", "").strip()

wr_df["player_norm"] = wr_df["player"].apply(normalize_name)
blitz_df["player_norm"] = blitz_df["player"].apply(normalize_name)

# Merge blitz data
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
    def_df[col] = def_df[col] / 100.0  # convert percentages to 0-1

# Merge matchups
wr_df = wr_df.merge(matchup_df, on="team", how="left")

# ------------------------
# Precompute league averages for deviation system
# ------------------------
league_avg_man = def_df["man_pct"].mean()
league_avg_zone = def_df["zone_pct"].mean()
league_avg_1high = def_df["onehigh_pct"].mean()
league_avg_2high = def_df["twohigh_pct"].mean()
league_avg_0high = def_df["zerohigh_pct"].mean()

# ------------------------
# Core Model Function
# ------------------------
def compute_model(
    wr_df,
    def_df,
    max_penalty=0.6,
    exponent=2,
    start_penalty=30,
    end_penalty=5,
    deviation_boost=0.25  # max ±25% influence
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
        # Blitz fallback
        blitz_ratio = row.get("yprr_blitz", np.nan)
        if pd.isna(blitz_ratio):
            blitz_ratio = base
        blitz_ratio /= base

        # ------------------------
        # Coverage & safety ratios
        man_ratio = row["yprr_man"] / base
        zone_ratio = row["yprr_zone"] / base
        onehigh_ratio = row["yprr_1high"] / base
        twohigh_ratio = row["yprr_2high"] / base
        zerohigh_ratio = row["yprr_0high"] / base

        man_pct = defense["man_pct"]
        zone_pct = defense["zone_pct"]
        onehigh_pct = defense["onehigh_pct"]
        twohigh_pct = defense["twohigh_pct"]
        zerohigh_pct = defense["zerohigh_pct"]

        # ------------------------
        # System A: main team weighting
        coverage_component = man_pct * man_ratio + zone_pct * zone_ratio
        total_coverage = man_pct + zone_pct
        safety_component = onehigh_pct * onehigh_ratio + twohigh_pct * twohigh_ratio + zerohigh_pct * zerohigh_ratio
        total_safety = onehigh_pct + twohigh_pct + zerohigh_pct
        if total_safety > 0:
            safety_component /= total_safety

        if total_coverage + total_safety > 0:
            systemA_ratio = (coverage_component * total_coverage + safety_component * total_safety) / (total_coverage + total_safety)
        else:
            systemA_ratio = (coverage_component + safety_component) / 2

        # ------------------------
        # System B: league deviation weighting
        coverage_dev = abs(man_pct - league_avg_man) + abs(zone_pct - league_avg_zone)
        safety_dev = abs(onehigh_pct - league_avg_1high) + abs(twohigh_pct - league_avg_2high) + abs(zerohigh_pct - league_avg_0high)

        if coverage_dev + safety_dev > 0:
            coverage_weight_dev = coverage_dev / (coverage_dev + safety_dev)
            safety_weight_dev = safety_dev / (coverage_dev + safety_dev)
        else:
            coverage_weight_dev = 0.5
            safety_weight_dev = 0.5

        systemB_ratio = coverage_component * coverage_weight_dev + safety_component * safety_weight_dev

        # ------------------------
        # Hybrid: final ratio with deviation boost
        final_ratio = systemA_ratio * (1 - deviation_boost) + systemB_ratio * deviation_boost

        # ------------------------
        # Compute adjusted YPRR & edge
        adjusted_yprr = base * ((final_ratio + blitz_ratio) / 2)
        raw_edge = (adjusted_yprr - base) / base
        edge_score = (raw_edge / 0.25) * 10

        # ------------------------
        # Route-share regression toward zero
        route_share = row.get("route_share", np.nan)
        if pd.isna(route_share):
            route_share = 0

        if route_share >= start_penalty:
            penalty = 0
        elif route_share <= end_penalty:
            penalty = max_penalty
        else:
            penalty = max_penalty * ((start_penalty - route_share) / (start_penalty - end_penalty)) ** exponent

        edge_score *= (1 - penalty)

        # ------------------------
        # Store both system contributions for visualization
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

    # Apply route-share filters
    if qualified_toggle_35:
        df = df[df["Route (%)"] >= 35]
    elif qualified_toggle_20:
        df = df[df["Route (%)"] >= 20]

    df = df.reindex(df["Edge"].abs().sort_values(ascending=False).index)
    df["Rk"] = range(1, len(df) + 1)

    return df

# ------------------------
# Edge color function
# ------------------------
def color_edge(val):
    if val > 20:
        return "color: darkgreen; font-weight: bold; text-align: center"
    elif 10 < val <= 20:
        return "color: green; font-weight: bold; text-align: center"
    elif 0 < val <= 10:
        return "color: lightgreen; font-weight: bold; text-align: center"
    elif -10 < val <= 0:
        return "color: lightcoral; font-weight: bold; text-align: center"
    elif -20 < val <= -10:
        return "color: red; font-weight: bold; text-align: center"
    else:
        return "color: darkred; font-weight: bold; text-align: center"

# ------------------------
# Run Model
# ------------------------
results = compute_model(wr_df, def_df)
if results.empty:
    st.warning("No players available after filtering.")
    st.stop()

# ------------------------
# Team Filter (Dropdown Style)
# ------------------------
st.sidebar.header("Team Filter")
team_options = sorted(results["Tm"].dropna().unique())

selected_teams = st.sidebar.multiselect(
    "Type or select team(s) to display (leave empty for all)",
    options=team_options
)

if selected_teams:
    results = results[results["Tm"].isin(selected_teams)]

# ------------------------
# Display Tables
# ------------------------
display_cols = [
    "Rk", "Player", "Tm", "Vs.", "Route (%)",
    "Base YPRR", "Adj. YPRR", "Matchup (+/-)", "Deviation", "Edge"
]

number_format = {
    "Edge": "{:.1f}",
    "Matchup (+/-)": "{:.1f}",
    "Deviation": "{:.1f}",
    "Route (%)": "{:.1f}",
    "Base YPRR": "{:.2f}",
    "Adj. YPRR": "{:.2f}"
}

st.subheader("Player Rankings")
st.dataframe(
    results[display_cols]
    .style
    .applymap(color_edge, subset=["Edge"])
    .format(number_format)
    .set_properties(**{'text-align': 'center'}, subset=[col for col in display_cols if col != 'Player'])
)

# ------------------------
# Targets & Fades thresholds
# ------------------------
min_edge = 15
min_routes = 30  # percentage

st.subheader("Targets")
st.markdown(f"*Showing players with Edge ≥ {min_edge} and Route Share ≥ {min_routes}%*")
targets = results[
    (results["Edge"] >= min_edge) &
    (results["Route (%)"] >= min_routes)
]

st.dataframe(
    targets[display_cols]
    .style
    .applymap(color_edge, subset=["Edge"])
    .format(number_format)
    .set_properties(**{'text-align': 'center'}, subset=[col for col in display_cols if col != 'Player'])
)

st.subheader("Fades")
st.markdown(f"*Showing players with Edge ≤ -{min_edge} and Route Share ≥ {min_routes}%*")
fades = results[
    (results["Edge"] <= -min_edge) &
    (results["Route (%)"] >= min_routes)
].sort_values("Edge")

st.dataframe(
    fades[display_cols]
    .style
    .applymap(color_edge, subset=["Edge"])
    .format(number_format)
    .set_properties(**{'text-align': 'center'}, subset=[col for col in display_cols if col != 'Player'])
)

# ------------------------
# Deviation boost bar plot
# ------------------------
st.subheader("Deviation Boost Impact")
st.markdown(
    "Bar plot shows the contribution of Matchup (+/-) vs Deviation on each player's Edge."
)

if not results.empty:
    plot_df = results.copy()
    plot_df = plot_df.sort_values("Edge", ascending=False).head(30)  # top 30 players

    plot_df_melt = plot_df.melt(
        id_vars=["Player"],
        value_vars=["Matchup (+/-)", "Deviation"],
        var_name="Component",
        value_name="Edge_Contribution"
    )

    chart = alt.Chart(plot_df_melt).mark_bar().encode(
        x=alt.X('Player', sort=None),
        y='Edge_Contribution',
        color=alt.Color('Component', scale=alt.Scale(domain=["Matchup (+/-)","Deviation"], range=["#4caf50","#ff9800"])),
        tooltip=['Player','Component','Edge_Contribution']
    ).properties(width=800, height=400)

    st.altair_chart(chart, use_container_width=True)

# ------------------------
# Column Descriptions
# ------------------------
st.subheader("Column Descriptions")
st.markdown("""
- **Rk**: Player's rank based on absolute Edge.
- **Matchup (+/-)**: Projection based purely on the team's coverage and safety tendencies.
- **Deviation**: Boost or detract based on how unique the team's defensive tendencies are relative to the league.
- **Edge**: Final score after combining Matchup (+/-), Deviation, and route-share regression.
- **Route (%)**: Percent of team routes run by the player.
- **Base YPRR / Adj. YPRR**: Yards per route run, before and after matchup adjustments.
- **Vs.**: Opponent team.
- **Tm**: Player's team.
""")


