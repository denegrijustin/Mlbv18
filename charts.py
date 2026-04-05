from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

# Disable zoom, pan, and the plotly toolbar on all charts for a clean
# read-only display. Users can still hover for tooltips.
_NO_ZOOM_CONFIG = {
    'scrollZoom': False,
    'displayModeBar': False,
    'staticPlot': False,
    'modeBarButtonsToRemove': [
        'zoom2d', 'pan2d', 'select2d', 'lasso2d',
        'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d',
        'zoom3d', 'pan3d', 'orbitRotation', 'tableRotation',
        'handleDrag3d', 'resetCameraDefault3d', 'resetCameraLastSave3d',
        'hoverClosest3d', 'hoverClosestCartesian', 'hoverCompareCartesian',
        'zoomInGeo', 'zoomOutGeo', 'resetGeo', 'hoverClosestGeo',
        'hoverClosestGl2d', 'hoverClosestPie', 'toggleHover',
        'resetViews', 'toggleSpikelines', 'resetViewMapbox',
    ],
}


def render_schedule_chart(schedule_df: pd.DataFrame) -> None:
    if schedule_df.empty:
        st.info('No schedule data available for charting.')
        return
    df = schedule_df.copy()
    df['matchup'] = df['away'] + ' at ' + df['home']
    fig = px.bar(df, x='matchup', y=['away_score', 'home_score'], barmode='group', title="Today's Score Snapshot")
    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
        dragmode=False,
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
    )
    st.plotly_chart(fig, use_container_width=True, config=_NO_ZOOM_CONFIG)


def render_recent_trend_chart(recent_games_df: pd.DataFrame) -> None:
    if recent_games_df.empty:
        st.info('No completed recent games are available yet.')
        return
    df = recent_games_df.copy()
    fig = px.line(df, x='Date', y=['Team Runs', 'Opp Runs'], markers=True, title='Recent Runs Trend')
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        dragmode=False,
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
    )
    st.plotly_chart(fig, use_container_width=True, config=_NO_ZOOM_CONFIG)


def render_run_diff_chart(recent_games_df: pd.DataFrame) -> None:
    if recent_games_df.empty:
        return
    fig = px.bar(recent_games_df, x='Date', y='Run Diff', color='Result', title='Recent Game Run Differential')
    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
        dragmode=False,
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
    )
    st.plotly_chart(fig, use_container_width=True, config=_NO_ZOOM_CONFIG)


def render_rolling_chart(rolling_df: pd.DataFrame) -> None:
    if rolling_df.empty:
        st.info('Not enough completed games for rolling trend lines yet.')
        return
    fig = px.line(rolling_df, x='Date', y=['Runs 3', 'Runs 5', 'Diff 3', 'Diff 5'], markers=True, title='Rolling Team Trend Lines')
    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
        dragmode=False,
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
    )
    st.plotly_chart(fig, use_container_width=True, config=_NO_ZOOM_CONFIG)


def render_pitch_mix_chart(pitch_mix_df: pd.DataFrame) -> None:
    if pitch_mix_df.empty:
        st.info('Pitch-mix data is not available yet.')
        return
    fig = px.bar(pitch_mix_df, x='Pitch Type', y='Usage %', color='Success', title='Pitch Type Usage')
    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
        dragmode=False,
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
    )
    st.plotly_chart(fig, use_container_width=True, config=_NO_ZOOM_CONFIG)


def render_statcast_scatter(batter_df: pd.DataFrame) -> None:
    if batter_df.empty:
        st.info('Exit velocity data is not available yet.')
        return
    fig = px.scatter(
        batter_df,
        x='Avg EV',
        y='Hard Hit %',
        size='PA',
        color='Grade',
        hover_name='Batter',
        hover_data=['xwOBA', 'XBH Rate %', 'Whiff %'],
        title='Batter Quality of Contact',
    )
    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
        dragmode=False,
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
    )
    st.plotly_chart(fig, use_container_width=True, config=_NO_ZOOM_CONFIG)
