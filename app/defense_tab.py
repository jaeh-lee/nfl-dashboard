"""The Pass defense tab: which defenses allow the most passing yards, and each defense week by week."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BLUE, RED = "royalblue", "crimson"


def defense_chart(dd: pd.DataFrame, defense: str, full, title: str | None = None,
                  last_n: int | None = None) -> go.Figure:
    """Passing yards one defense allowed in each regular season game of a season.

    Bars are the yards allowed (red when above what that offense usually gets, blue when below),
    diamonds are what that offense gets in its other games, and the dashed line is the league average.
    dd must hold a single season. When last_n is given, a dotted line marks the average of the defense's
    last last_n games.
    """
    d = dd[dd["defense"] == defense].sort_values("gameday")
    league = dd["passing_yards"].mean()
    labels = [f"Wk {int(w)} {o}" for w, o in zip(d["week"], d["offense"])]
    detail = [
        f"{(q if isinstance(q, str) else 'unknown QB')} ({full(o)})<br>{int(a)} attempts, {int(y)} yards, {p:.1f} per attempt"
        for q, o, a, y, p in zip(d["opposing_qb"], d["offense"], d["attempts"], d["passing_yards"], d["ypa"])
    ]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=d["passing_yards"], name="Passing yards allowed", opacity=0.7,
        marker_color=[RED if v > 0 else BLUE for v in d["vs_offense_avg"].fillna(0)],
        text=d["passing_yards"].astype(int), textposition="outside", hovertext=detail, hoverinfo="text",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=d["offense_avg_other"], mode="markers", name="That offense's usual (other games)",
        marker=dict(symbol="diamond", size=11, color="black"),
    ))
    fig.add_hline(y=league, line_dash="dash", line_color="gray",
                  annotation_text=f"League average {league:.0f}", annotation_position="top left")
    if last_n and len(d):
        recent_avg = d["passing_yards"].tail(last_n).mean()
        fig.add_hline(y=recent_avg, line_dash="dot", line_color="black",
                      annotation_text=f"Last {min(last_n, len(d))} games avg {recent_avg:.0f}",
                      annotation_position="top right")
    fig.update_layout(title=title or f"{full(defense)}: passing yards allowed by game", height=420,
                      yaxis_title="Passing yards allowed", yaxis_rangemode="tozero",
                      legend=dict(orientation="h", yanchor="top", y=-0.25), margin=dict(t=60, b=40))
    return fig


def render_defense_tab(pass_def: pd.DataFrame, season: int, full) -> None:
    """Draw the tab for one season."""
    if pass_def.empty:
        st.info("No pass defense data yet. It is created by `python -m src.qb predict`, which the refresh "
                "workflow runs after each data update.")
        return
    dd = pass_def[pass_def["season"] == season]
    if dd.empty:
        st.info(f"No regular season games have been played yet in {season}.")
        return

    st.subheader(f"{season} pass defense: passing yards allowed (regular season)")
    league = dd["passing_yards"].mean()
    g = dd.groupby("defense")
    table = pd.DataFrame({
        "games": g.size(),
        "yards": g["passing_yards"].mean(),
        "attempts": g["attempts"].mean(),
        "ypa": g["passing_yards"].sum() / g["attempts"].sum(),
        "last4": dd.sort_values("gameday").groupby("defense").tail(4).groupby("defense")["passing_yards"].mean(),
        "adj": g["vs_offense_avg"].mean(),
    })
    table["vs_league"] = table["yards"] - league
    table = table.sort_values("yards", ascending=False)
    table.insert(0, "rank", range(1, len(table) + 1))
    show = table.reset_index()
    show["defense"] = show["defense"].map(full)
    st.dataframe(
        show[["rank", "defense", "games", "yards", "vs_league", "attempts", "ypa", "last4", "adj"]],
        hide_index=True, width="stretch",
        column_config={
            "rank": st.column_config.NumberColumn("Rank (1 = allows most)", format="%d"),
            "defense": "Defense",
            "games": st.column_config.NumberColumn("Games", format="%d"),
            "yards": st.column_config.NumberColumn("Yards allowed/game", format="%.1f"),
            "vs_league": st.column_config.NumberColumn("vs league avg", format="%+.1f"),
            "attempts": st.column_config.NumberColumn("Attempts faced/game", format="%.1f"),
            "ypa": st.column_config.NumberColumn("Yards per attempt", format="%.2f"),
            "last4": st.column_config.NumberColumn("Last 4 games avg", format="%.1f"),
            "adj": st.column_config.NumberColumn("Vs offenses' usual", format="%+.1f"),
        },
    )
    st.caption(
        f"League average: {league:.1f} passing yards per game. 'Vs offenses' usual' is the average of "
        "(yards allowed minus what that offense gets in its other games), so a negative number means the defense "
        "tends to hold offenses below their norm, which corrects for the quality of the offenses faced. Early "
        "in a season these averages rest on few games and move a lot: across 2021 to 2025, a defense's first "
        "9 games and last 8 games correlated only about 0.19."
    )

    st.subheader("One defense, game by game")
    options = list(show["defense"])
    codes = dict(zip(show["defense"], table.index))
    pick = st.selectbox("Defense", options, key="defense_pick")
    code = codes[pick]
    st.plotly_chart(defense_chart(dd, code, full), width="stretch")

    log = dd[dd["defense"] == code].sort_values("gameday")
    st.dataframe(pd.DataFrame({
        "Week": log["week"].astype(int).values,
        "Date": log["gameday"].values,
        "Offense": log["offense"].map(full).values,
        "Opposing QB": log["opposing_qb"].values,
        "Home/away": log["defense_location"].values,
        "Attempts": log["attempts"].astype(int).values,
        "Yards allowed": log["passing_yards"].astype(int).values,
        "Offense's usual": log["offense_avg_other"].round(0).values,
        "Difference": log["vs_offense_avg"].round(0).values,
    }), hide_index=True, width="stretch")
    st.caption("Bars are red when the defense allowed more than that offense usually gets and blue when less. "
               "Diamonds mark what that offense gets per game in its other games that season.")