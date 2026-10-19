"""Streamlit app: standings and team trends from data/nfl.db.

Run from the repo root:  streamlit run app/main.py
The app only reads the database. Refresh it with `python -m src.load` and `python -m src.metrics`.
"""

import sys
from pathlib import Path

# Streamlit puts the app/ folder on sys.path, not the repo root, so add the root
# here to be able to import from src.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.db import DB_PATH, get_connection
from src.metrics import PYTH_EXP, WINDOW, read_metrics, season_summary
from src.queries import read_games, read_team_names

st.set_page_config(page_title="NFL Dashboard", layout="wide")

BLUE, RED, GRAY = "royalblue", "crimson", "gray"
SCOPES = {"Regular season": "regular", "Playoffs": "playoffs", "Both": "both"}


# ---------------------------------------------------------------- data loading

@st.cache_data
def load_all(db_mtime: float):
    """Read metrics, all games and team names. db_mtime is only a cache key: when the file changes, re-read."""
    conn = get_connection()
    try:
        return read_metrics(conn), read_games(conn), read_team_names(conn)
    finally:
        conn.close()


# Team abbreviation to full name. Filled after loading, and used everywhere a team is shown.
NAMES: dict[str, str] = {}


def full(abbr: str) -> str:
    """Full team name, or the abbreviation itself if the teams table has no entry."""
    return NAMES.get(abbr, abbr)


def filter_games(m: pd.DataFrame, season: int, scope: str, team: str | None = None) -> pd.DataFrame:
    """Rows for one season, filtered to regular season, playoffs or both, optionally one team."""
    d = m[m["season"] == season]
    if scope == "regular":
        d = d[d["is_playoff"] == 0]
    elif scope == "playoffs":
        d = d[d["is_playoff"] == 1]
    if team:
        d = d[d["team"] == team]
    return d.sort_values("game_number")


def add_labels(d: pd.DataFrame) -> pd.DataFrame:
    """Add display columns used by the tooltips and the game log."""
    d = d.copy()
    d["opponent"] = d["opponent"].map(full)
    d["score"] = d["points_for"].astype(str) + "-" + d["points_against"].astype(str)
    d["result"] = d["win"].map({1.0: "W", 0.5: "T", 0.0: "L"})
    return d


# ---------------------------------------------------------------- chart helpers

# Columns every tooltip can use: 0 date, 1 home/away, 2 opponent, 3 score, 4 result.
HOVER_COLS = ["gameday", "location", "opponent", "score", "result"]
HOVER_HEAD = "Week %{x} (%{customdata[0]})<br>%{customdata[1]} vs %{customdata[2]}: %{customdata[3]} (%{customdata[4]})"


def style(fig: go.Figure, title: str, y_title: str, y_range=None) -> go.Figure:
    """Shared layout: one tick per week, titles, and a legend under the plot."""
    fig.update_layout(
        title=title, height=400, hovermode="closest",
        legend=dict(orientation="h", yanchor="top", y=-0.2),
        margin=dict(t=60, b=40),
    )
    fig.update_xaxes(title_text="Week", dtick=1)
    fig.update_yaxes(title_text=y_title, range=y_range)
    return fig


def points_chart(d: pd.DataFrame, team: str) -> go.Figure:
    """Points scored and allowed by week, home games as circles and away games as diamonds."""
    fig = go.Figure()
    symbols = d["location"].map({"home": "circle", "away": "diamond"}).tolist()

    for col, color, name in [("points_for", BLUE, "scored"), ("points_against", RED, "allowed")]:
        fig.add_trace(go.Scatter(
            x=d["week"], y=d[col], mode="lines+markers", name=f"Points {name}",
            line=dict(color=color), marker=dict(symbol=symbols, size=11),
            customdata=d[HOVER_COLS],
            hovertemplate=HOVER_HEAD + f"<br>Points {name}: " + "%{y}<extra></extra>",
        ))

    # Legend-only entries that explain the marker shapes (no data, so nothing is drawn).
    for sym, label in [("circle", "Home game"), ("diamond", "Away game")]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=label, hoverinfo="skip",
                                 marker=dict(symbol=sym, size=11, color=GRAY)))

    # Mean (dashed) and median (dotted) over the games shown, as flat lines.
    x_min, x_max = d["week"].min(), d["week"].max()
    for col, color, name in [("points_for", BLUE, "scored"), ("points_against", RED, "allowed")]:
        for stat, dash, val in [("mean", "dash", d[col].mean()), ("median", "dot", d[col].median())]:
            fig.add_trace(go.Scatter(
                x=[x_min, x_max], y=[val, val], mode="lines", line=dict(color=color, dash=dash, width=1.5),
                name=f"{name} {stat}: {val:.1f}", hovertemplate=f"{name} {stat}: {val:.1f}<extra></extra>",
            ))
    return style(fig, f"{full(team)}: points scored vs allowed by week", "Points", [0, None])


def rolling_win_chart(d: pd.DataFrame) -> go.Figure:
    """Rolling win % over the last WINDOW games against the season-to-date win %."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["week"], y=d["roll_win_pct"], mode="lines+markers", name=f"Rolling win % ({WINDOW} games)",
        line=dict(color=BLUE), customdata=d[HOVER_COLS],
        hovertemplate=HOVER_HEAD + "<br>Rolling win %: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=d["week"], y=d["season_win_pct"], mode="lines+markers", name="Season win % to date",
        line=dict(color=GRAY, dash="dash"), customdata=d[HOVER_COLS],
        hovertemplate=HOVER_HEAD + "<br>Season win %: %{y:.3f}<extra></extra>",
    ))
    return style(fig, "Rolling win % vs season win %", "Win %", [0, 1.05])


def rolling_margin_chart(d: pd.DataFrame) -> go.Figure:
    """Single-game margin as bars and the rolling average margin as a line."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=d["week"], y=d["margin"], name="Game margin", opacity=0.35,
        marker_color=[BLUE if v > 0 else RED for v in d["margin"]], customdata=d[HOVER_COLS],
        hovertemplate=HOVER_HEAD + "<br>Margin: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=d["week"], y=d["roll_margin"], mode="lines+markers", name=f"Rolling margin ({WINDOW} games)",
        line=dict(color="black"), customdata=d[HOVER_COLS],
        hovertemplate=HOVER_HEAD + "<br>Rolling margin: %{y:.1f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_width=1, line_color=GRAY)
    return style(fig, "Point margin: each game and rolling average", "Points")


def adjusted_chart(d: pd.DataFrame) -> go.Figure:
    """Opponent-adjusted margin per game and rolling, with the raw rolling margin for comparison."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=d["week"], y=d["adj_margin"], name="Adjusted game margin", opacity=0.35,
        marker_color=[BLUE if v > 0 else RED for v in d["adj_margin"]],
        customdata=d[HOVER_COLS + ["opp_prior_avg_margin", "margin"]],
        hovertemplate=(HOVER_HEAD + "<br>Margin: %{customdata[6]}"
                       "<br>Opponent's average margin before: %{customdata[5]:.1f}"
                       "<br>Adjusted margin: %{y:.1f}<extra></extra>"),
    ))
    fig.add_trace(go.Scatter(
        x=d["week"], y=d["roll_adj_margin"], mode="lines+markers", name=f"Rolling adjusted margin ({WINDOW} games)",
        line=dict(color="black"), customdata=d[HOVER_COLS],
        hovertemplate=HOVER_HEAD + "<br>Rolling adjusted margin: %{y:.1f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=d["week"], y=d["roll_margin"], mode="lines", name=f"Rolling raw margin ({WINDOW} games)",
        line=dict(color=GRAY, dash="dash"), customdata=d[HOVER_COLS],
        hovertemplate=HOVER_HEAD + "<br>Rolling raw margin: %{y:.1f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_width=1, line_color=GRAY)
    return style(fig, "Opponent-adjusted margin", "Points")


def pythag_chart(d: pd.DataFrame) -> go.Figure:
    """Actual season win % against the Pythagorean win % implied by points scored and allowed."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["week"], y=d["season_win_pct"], mode="lines+markers", name="Actual win %",
        line=dict(color=BLUE), customdata=d[HOVER_COLS],
        hovertemplate=HOVER_HEAD + "<br>Actual win %: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=d["week"], y=d["pythag_win_pct"], mode="lines+markers", name=f"Pythagorean win % (exp {PYTH_EXP})",
        line=dict(color=RED, dash="dash"), customdata=d[HOVER_COLS],
        hovertemplate=HOVER_HEAD + "<br>Pythagorean win %: %{y:.3f}<extra></extra>",
    ))
    return style(fig, "Actual vs Pythagorean win % (season to date)", "Win %", [0, 1.05])


# ---------------------------------------------------------------- weekly view helpers

PLAYOFF_ROUNDS = {"WC": "Wild Card", "DIV": "Divisional", "CON": "Conference Championship", "SB": "Super Bowl"}


def week_label(week_games: pd.DataFrame, week: int) -> str:
    """'Week 7' for regular season weeks, or the playoff round name."""
    types = set(week_games.loc[week_games["week"] == week, "game_type"])
    if types == {"REG"}:
        return f"Week {week}"
    first = sorted(types - {"REG"})[0] if types - {"REG"} else "REG"
    return PLAYOFF_ROUNDS.get(first, f"Week {week}")


def default_week(week_games: pd.DataFrame) -> int:
    """The first week that still has unplayed games, or the last week if everything is played."""
    open_weeks = week_games.loc[~week_games["completed"], "week"]
    return int(open_weeks.min()) if len(open_weeks) else int(week_games["week"].max())


def count_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Games, completed and remaining, grouped by a column in order of first appearance."""
    out = df.groupby(by, sort=False).agg(
        games=("game_id", "size"), completed=("completed", "sum"),
    )
    out["completed"] = out["completed"].astype(int)
    out["remaining"] = out["games"] - out["completed"]
    return out


def record_text(wins: int, losses: int, ties: int) -> str:
    """'10-4' or '10-3-1' when there are ties."""
    return f"{wins}-{losses}" + (f"-{ties}" if ties else "")


def records_entering(season_games: pd.DataFrame):
    """Regular season record each team brought into each game, plus each team's total so far.

    Returns (before, totals):
      before: {(game_id, team): record text} for regular season games, counting only earlier games.
      totals: {team: record text} through the last completed regular season game. Playoff games use
              this, because every regular season game comes before them.
    """
    reg = season_games[season_games["game_type"] == "REG"]
    done = reg["home_score"].notna() & reg["away_score"].notna()

    # One row per team per game, with a 1 in the column for the result (unplayed games are all 0).
    sides = []
    for side, other in (("home", "away"), ("away", "home")):
        sides.append(pd.DataFrame({
            "game_id": reg["game_id"], "team": reg[f"{side}_team"],
            "gameday": reg["gameday"], "gametime": reg["gametime"],
            "win": (done & (reg[f"{side}_score"] > reg[f"{other}_score"])).astype(int),
            "loss": (done & (reg[f"{side}_score"] < reg[f"{other}_score"])).astype(int),
            "tie": (done & (reg[f"{side}_score"] == reg[f"{other}_score"])).astype(int),
        }))
    long = pd.concat(sides).sort_values(["team", "gameday", "gametime", "game_id"])

    # Cumulative results through each game, minus that game's own result = the record entering it.
    by_team = long.groupby("team")
    for col in ("win", "loss", "tie"):
        long[col + "_before"] = by_team[col].cumsum() - long[col]

    before = {
        (r.game_id, r.team): record_text(r.win_before, r.loss_before, r.tie_before)
        for r in long.itertuples()
    }
    sums = by_team[["win", "loss", "tie"]].sum()
    totals = {team: record_text(r.win, r.loss, r.tie) for team, r in sums.iterrows()}
    return before, totals

# ---------------------------------------------------------------- load and sidebar

if not DB_PATH.exists():
    st.error("Database not found at data/nfl.db. From the repo root run `python -m src.load` "
             "and then `python -m src.metrics`.")
    st.stop()

try:
    metrics, games, team_names = load_all(DB_PATH.stat().st_mtime)
except Exception as exc:  # for example, the metrics table has not been created yet
    st.error(f"Could not read the database tables ({exc}). Run `python -m src.load` and "
             "`python -m src.metrics` from the repo root.")
    st.stop()

NAMES.update(team_names)

st.title("NFL Analytics Dashboard")

st.sidebar.header("Filters")
season = int(st.sidebar.selectbox("Season", sorted(games["season"].unique(), reverse=True)))
scope_label = st.sidebar.radio("Games", list(SCOPES), index=0)
scope = SCOPES[scope_label]

# Team choices come from games that have been played, because the trend tabs need metrics.
pool = filter_games(metrics, season, scope)
has_metrics = not pool.empty
team = None
if has_metrics:
    team = st.sidebar.selectbox(
        "Team (Team trends and Game log)", sorted(pool["team"].unique(), key=full), format_func=full
    )

played_dates = metrics["gameday"].max() if len(metrics) else "none yet"
st.sidebar.caption(f"Latest game in data: {played_dates}")
st.sidebar.caption(f"Rolling window: {WINDOW} games. Pythagorean exponent: {PYTH_EXP}. "
                   "Playoffs are wild card, divisional, conference and Super Bowl games.")

tab_week, tab_standings, tab_trends, tab_log = st.tabs(
    ["Weekly games", "Standings", "Team trends", "Game log"]
)

# ---------------------------------------------------------------- weekly games

with tab_week:
    season_games = games[games["season"] == season]
    sg = season_games
    if scope == "regular":
        sg = sg[sg["game_type"] == "REG"]
    elif scope == "playoffs":
        sg = sg[sg["game_type"] != "REG"]
    sg = sg.assign(completed=sg["home_score"].notna() & sg["away_score"].notna())

    if sg.empty:
        st.info(f"No {scope_label.lower()} games are scheduled yet for {season}.")
    else:
        # Season progress for the selected filter.
        total, done = len(sg), int(sg["completed"].sum())
        st.subheader(f"{season} season progress ({scope_label.lower()})")
        a, b, c = st.columns(3)
        a.metric("Games", total)
        b.metric("Completed", done)
        c.metric("Remaining", total - done)
        st.progress(done / total, text=f"{done} of {total} games completed")

        with st.expander("All weeks at a glance"):
            by_week = count_table(sg.sort_values(["week", "gameday"]), "week").reset_index()
            by_week.insert(0, "round", [week_label(sg, w) for w in by_week["week"]])
            st.dataframe(
                by_week.drop(columns="week"), hide_index=True, width="stretch",
                column_config={"round": "Week", "games": "Games", "completed": "Completed",
                               "remaining": "Remaining"},
            )

        # Week picker, defaulting to the first week that still has games to play.
        weeks = sorted(sg["week"].unique())
        pick = st.selectbox(
            "Week", weeks, index=weeks.index(default_week(sg)),
            format_func=lambda w: week_label(sg, w), key=f"week_{season}_{scope}",
        )
        wk = sg[sg["week"] == pick].sort_values(["gameday", "gametime"])

        st.subheader(f"{week_label(sg, pick)}")

        # Teams without a game this week (regular season weeks only).
        if set(wk["game_type"]) == {"REG"}:
            teams_all = set(season_games["home_team"]) | set(season_games["away_team"])
            playing = set(wk["home_team"]) | set(wk["away_team"])
            on_bye = sorted((full(t) for t in teams_all - playing))
            st.markdown("**On bye:** " + (", ".join(on_bye) if on_bye else "none"))

        # Regular season record each team brought into its game.
        before, totals = records_entering(season_games)

        def entering(game_id: str, team: str, game_type: str) -> str:
            if game_type == "REG":
                return before.get((game_id, team), "")
            return totals.get(team, "")  # playoff games: the full regular season record

        # The games themselves, in kickoff order.
        table = pd.DataFrame({
            "Day": wk["weekday"].values,
            "Date": wk["gameday"].values,
            "Kickoff (ET)": wk["gametime"].values,
            "Away": wk["away_team"].map(full).values,
            "Away record": [entering(g, t, k) for g, t, k in zip(wk["game_id"], wk["away_team"], wk["game_type"])],
            "Away score": wk["away_score"].astype("Int64").values,
            "Home": wk["home_team"].map(full).values,
            "Home record": [entering(g, t, k) for g, t, k in zip(wk["game_id"], wk["home_team"], wk["game_type"])],
            "Home score": wk["home_score"].astype("Int64").values,
            "Status": ["Final" if x else "Not played yet" for x in wk["completed"]],
        })
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption("Records are regular season records (wins-losses, or wins-losses-ties) before that game. "
                   "Status reflects the last data refresh (Monday, Tuesday and Friday). A game that has "
                   "finished since then shows as not played yet until the next refresh.")

# ---------------------------------------------------------------- standings

with tab_standings:
    if not has_metrics:
        st.info(f"No {scope_label.lower()} games have been played yet in {season}.")
    else:
        st.subheader(f"{season} standings ({scope_label.lower()})")
        summary = season_summary(metrics, season, scope)

        table = summary.reset_index()
        table["team"] = table["team"].map(full)
        table.insert(0, "rank", range(1, len(table) + 1))
        table.insert(2, "record", [
            f"{r.wins}-{r.losses}" + (f"-{r.ties}" if r.ties else "") for r in table.itertuples()
        ])
        cols = ["rank", "team", "record", "games", "win_pct", "points_for", "points_against",
                "margin_per_game", "pythag_win_pct", "home_win_pct", "home_margin_per_game",
                "away_win_pct", "away_margin_per_game"]

        st.dataframe(
            table[cols], hide_index=True, width="stretch",
            column_config={
                "rank": st.column_config.NumberColumn("Rank", format="%d"),
                "team": "Team",
                "record": "Record",
                "games": st.column_config.NumberColumn("Games", format="%d"),
                "win_pct": st.column_config.NumberColumn("Win %", format="%.3f"),
                "points_for": st.column_config.NumberColumn("PF", format="%d"),
                "points_against": st.column_config.NumberColumn("PA", format="%d"),
                "margin_per_game": st.column_config.NumberColumn("Margin/game", format="%.1f"),
                "pythag_win_pct": st.column_config.NumberColumn("Pythag win %", format="%.3f"),
                "home_win_pct": st.column_config.NumberColumn("Home win %", format="%.3f"),
                "home_margin_per_game": st.column_config.NumberColumn("Home margin/game", format="%.1f"),
                "away_win_pct": st.column_config.NumberColumn("Away win %", format="%.3f"),
                "away_margin_per_game": st.column_config.NumberColumn("Away margin/game", format="%.1f"),
            },
        )
        st.caption("Sorted by win %, then margin per game. Ties count as half a win. "
                   "All columns are computed only from the games in the selected filter.")

# ---------------------------------------------------------------- team trends

with tab_trends:
    if not has_metrics:
        st.info(f"No {scope_label.lower()} games have been played yet in {season}.")
    else:
        d = add_labels(filter_games(metrics, season, scope, team))
        st.subheader(f"{full(team)}: {season} ({scope_label.lower()}), {len(d)} games")
        st.caption("Rolling and adjusted values are computed across the team's whole season in game order. "
                   "The filter chooses which games are shown.")

        st.plotly_chart(points_chart(d, team), width="stretch")
        with st.expander("How to read median vs mean"):
            st.markdown(
                "| Series | Median above mean | Mean above median |\n"
                "| --- | --- | --- |\n"
                "| Points scored | Mostly solid scoring, 1-2 very weak games | Mostly modest scoring, 1-2 explosive games |\n"
                "| Points allowed | Usually gives up a lot, 1-2 dominant games | Usually holds opponents down, 1-2 blowout losses |\n"
                "| What it means | Left-skewed: a few very low games pull the mean down | Right-skewed: a few very high games pull the mean up |\n\n"
                "With only a handful of games, one result can swing the mean by several points."
            )

        left, right = st.columns(2)
        left.plotly_chart(rolling_win_chart(d), width="stretch")
        right.plotly_chart(rolling_margin_chart(d), width="stretch")

        left, right = st.columns(2)
        left.plotly_chart(adjusted_chart(d), width="stretch")
        right.plotly_chart(pythag_chart(d), width="stretch")
        st.caption("Adjusted margin = game margin plus the opponent's average margin before that game. "
                   "Beating a strong opponent is credited and beating a weak one is discounted. "
                   "Week 1 games are adjusted against zero because opponents have no history yet.")

# ---------------------------------------------------------------- game log

with tab_log:
    if not has_metrics:
        st.info(f"No {scope_label.lower()} games have been played yet in {season}.")
    else:
        st.subheader(f"{full(team)} game log: {season} ({scope_label.lower()})")
        show_pre = st.checkbox("Show pre-game values (what a model would know before kickoff)")

        log = add_labels(filter_games(metrics, season, scope, team))
        cols = ["week", "game_type", "gameday", "location", "opponent", "score", "result", "margin",
                "roll_win_pct", "roll_margin", "opp_prior_avg_margin", "adj_margin", "roll_adj_margin",
                "season_win_pct", "pythag_win_pct"]
        if show_pre:
            cols += ["pre_roll_win_pct", "pre_roll_margin", "pre_roll_adj_margin",
                     "pre_season_win_pct", "pre_pythag_win_pct"]

        st.dataframe(log[cols].round(3), hide_index=True, width="stretch")