"""The QB projections tab: projected passing yards for the next week and each QB's recent games."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BLUE, RED, GRAY = "royalblue", "crimson", "gray"


def render_qb_tab(qb_pred: pd.DataFrame, qb_games: pd.DataFrame, games: pd.DataFrame, full, week_label) -> None:
    """Draw the tab. `full` turns a team code into a full name, and `week_label` names a week."""
    if qb_pred.empty:
        st.info("No quarterback projections yet. They are created by `python -m src.qb predict`, "
                "which the refresh workflow runs after each data update.")
        return

    latest = qb_pred.sort_values(["season", "week"]).iloc[-1]
    pw = qb_pred[(qb_pred["season"] == latest["season"]) & (qb_pred["week"] == latest["week"])].copy()
    week_games = games[(games["season"] == latest["season"]) & (games["week"] == latest["week"])]
    st.subheader(f"{int(latest['season'])} {week_label(week_games, int(latest['week']))}: "
                 "quarterback passing yard projections")

    table = pd.DataFrame({
        "Quarterback": pw["player_name"].values,
        "Team": pw["team"].map(full).values,
        "Opponent": pw["opponent_team"].map(full).values,
        "Home/away": pw["location"].values,
        "Date": pw["gameday"].values,
        "Kickoff (ET)": pw["gametime"].values,
        "Projected yards": pw["projected_yards"].round(0).values,
        "Likely range": [f"{lo:.0f} to {hi:.0f}" for lo, hi in zip(pw["low_yards"], pw["high_yards"])],
        "Projected attempts": pw["projected_attempts"].round(0).values,
        "Last 8 starts avg": pw["qb_last_yards"].round(0).values,
        "Opponent allows/game": pw["opp_allowed_yards_pg"].round(0).values,
        "League avg/game": pw["league_yards_pg"].round(0).values,
        "Opponent rank": pw["opp_rank"].values,
        "Team implied total": pw["implied_total"].round(1).values,
    }).sort_values("Projected yards", ascending=False)
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(
        f"Model: ridge regression on the quarterback's recent form, the opponent's pass defense, home or away, "
        f"the team's implied total and spread, rest days and dome, trained on {int(pw['train_starts'].iloc[0])} "
        f"starts through {pw['trained_through'].iloc[0]}. The quarterback shown is the one who threw the most "
        "passes in his team's most recent game, so a change of starter or an injury is not reflected."
    )
    st.caption(
        "Likely range: the middle 80% of past misses of this model, so about 8 in 10 games land inside it. "
        "It is wide because a quarterback's yards vary a lot from game to game. Opponent rank 1 means the "
        "defense allows the most passing yards per game this season. Opponent and league averages are pulled "
        "toward each other early in the season, when few games have been played."
    )

    # ---- one quarterback's recent games against what each opponent usually allows
    st.subheader("Recent games against opponent averages")
    options = pw.sort_values("projected_yards", ascending=False)
    labels = {r.player_id: f"{r.player_name} ({r.team})" for r in options.itertuples()}
    pid = st.selectbox("Quarterback", list(labels), format_func=lambda x: labels[x], key="qb_pick")
    row = options[options["player_id"] == pid].iloc[0]

    recent = qb_games[qb_games["player_id"] == pid].sort_values("gameday").tail(10)
    if recent.empty:
        st.info("No earlier starts to show.")
        return
    log = pd.DataFrame({
        "Date": recent["gameday"].values,
        "Opponent": recent["opponent_team"].map(full).values,
        "Home/away": recent["location"].values,
        "Attempts": recent["attempts"].astype(int).values,
        "Passing yards": recent["passing_yards"].astype(int).values,
        "Opponent allows in its other games": recent["opp_avg_allowed_other"].round(0).values,
        "Difference": recent["vs_opp_avg"].round(0).values,
    })
    st.dataframe(log.iloc[::-1], hide_index=True, width="stretch")

    # Chart: yards each game with the opponent's usual allowed yards, then the projection with its range.
    x = [f"{d[5:]} {o}" for d, o in zip(recent["gameday"], recent["opponent_team"])]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=recent["passing_yards"], name="Passing yards", marker_color=BLUE, opacity=0.6))
    fig.add_trace(go.Scatter(x=x, y=recent["opp_avg_allowed_other"], mode="markers", name="Opponent allows (other games)",
                             marker=dict(symbol="diamond", size=11, color="black")))
    nxt = f"Next: {row['opponent_team']}"
    fig.add_trace(go.Bar(
        x=[nxt], y=[row["projected_yards"]], name="Projection", marker_color=RED, opacity=0.7,
        error_y=dict(type="data", symmetric=False, array=[row["high_yards"] - row["projected_yards"]],
                     arrayminus=[row["projected_yards"] - row["low_yards"]]),
    ))
    fig.add_trace(go.Scatter(x=[nxt], y=[row["opp_allowed_yards_pg"]], mode="markers", name="Opponent allows (season)",
                             marker=dict(symbol="diamond", size=11, color="black"), showlegend=False))
    fig.update_layout(title=f"{labels[pid]}: passing yards by game", height=420, yaxis_title="Passing yards",
                      legend=dict(orientation="h", yanchor="top", y=-0.25), margin=dict(t=60, b=40))
    st.plotly_chart(fig, width="stretch")
    st.caption("Bars are the quarterback's passing yards. Diamonds are the passing yards that opponent allowed per "
               "game in its other games that season, so a bar above its diamond beat that defense's usual. "
               "The red bar is the projection with its likely range.")