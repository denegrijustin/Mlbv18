from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from charts import (
    render_pitch_mix_chart,
    render_recent_trend_chart,
    render_rolling_chart,
    render_run_diff_chart,
    render_schedule_chart,
    render_statcast_scatter,
)
from data_helpers import (
    build_batter_grades_df,
    build_kpi_cards,
    build_live_box_df,
    build_pitch_mix_df,
    build_pitcher_grades_df,
    build_recent_games_df,
    build_schedule_table,
    build_standings_views,
    build_statcast_summary_df,
    build_summary_df,
    build_team_rolling_df,
    build_team_snapshot,
    build_trend_df,
    safe_team_row,
)
from formatting import coerce_float, coerce_int, stoplight
from mlb_api import (
    BASE_URL,
    HEADERS,
    build_schedule_df,
    build_season_df,
    choose_live_game_pk,
    get_live_summary,
    get_statcast_team_df,
    load_teams,
)

st.set_page_config(page_title='Live MLB Analytics Dashboard', layout='wide')

TIMEOUT = 25

_NO_ZOOM_CONFIG = {
    'scrollZoom': False,
    'displayModeBar': False,
    'staticPlot': False,
}


@st.cache_data(ttl=3600)
def cached_teams():
    return load_teams()


@st.cache_data(ttl=900)
def cached_daily(team_id: int, target_date: str):
    return build_schedule_df(team_id=team_id, target_date=target_date)


@st.cache_data(ttl=1800)
def cached_season(team_id: int, season: int, end_date: str):
    return build_season_df(team_id=team_id, season=season, end_date=end_date)


@st.cache_data(ttl=1800)
def cached_full_schedule(team_id: int, season: int):
    params = {
        'sportId': 1,
        'teamId': team_id,
        'startDate': f'{season}-01-01',
        'endDate': f'{season}-12-31',
        'gameType': 'R',
        'hydrate': 'team',
    }
    try:
        response = requests.get(f'{BASE_URL}/schedule', params=params, timeout=TIMEOUT, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        rows = []
        for day in data.get('dates', []):
            for g in day.get('games', []):
                teams = g.get('teams', {})
                away_team = (teams.get('away') or {}).get('team') or {}
                home_team = (teams.get('home') or {}).get('team') or {}
                status_obj = g.get('status') or {}
                rows.append({
                    'gamePk': coerce_int(g.get('gamePk'), 0),
                    'gameDate': str(g.get('gameDate', '')),
                    'officialDate': str(g.get('officialDate', '')),
                    'away': str(away_team.get('name', '')),
                    'home': str(home_team.get('name', '')),
                    'away_id': coerce_int(away_team.get('id'), 0),
                    'home_id': coerce_int(home_team.get('id'), 0),
                    'away_score': coerce_int((teams.get('away') or {}).get('score'), 0),
                    'home_score': coerce_int((teams.get('home') or {}).get('score'), 0),
                    'status': str(status_obj.get('detailedState') or status_obj.get('abstractGameState') or 'Unknown'),
                    'abstract_status': str(status_obj.get('abstractGameState') or 'Unknown'),
                })
        df = pd.DataFrame(rows)
        if df.empty:
            return df, None
        return df.sort_values(['officialDate', 'gameDate', 'gamePk']).reset_index(drop=True), None
    except Exception as exc:
        return pd.DataFrame(), str(exc)


@st.cache_data(ttl=1800)
def cached_standings(season: int):
    try:
        response = requests.get(
            f'{BASE_URL}/standings',
            params={'leagueId': '103,104', 'season': season, 'standingsTypes': 'regularSeason'},
            timeout=TIMEOUT,
            headers=HEADERS,
        )
        response.raise_for_status()
        data = response.json()
        rows = []
        for rec in data.get('records', []):
            division = rec.get('division') or {}
            league = rec.get('league') or {}
            for tr in rec.get('teamRecords', []):
                team = tr.get('team') or {}
                rows.append({
                    'team_id': coerce_int(team.get('id'), 0),
                    'team': str(team.get('name', '')),
                    'wins': coerce_int(tr.get('wins'), 0),
                    'losses': coerce_int(tr.get('losses'), 0),
                    'win_pct': coerce_float(tr.get('winningPercentage'), 0.0),
                    'games_back': coerce_float(tr.get('gamesBack'), 0.0),
                    'wc_games_back': coerce_float(tr.get('wildCardGamesBack'), 0.0),
                    'division_rank': coerce_int(tr.get('divisionRank'), 99),
                    'wild_card_rank': coerce_int(tr.get('wildCardRank'), 99),
                    'division_id': coerce_int(division.get('id'), 0),
                    'division_name': str(division.get('name', '')),
                    'league_id': coerce_int(league.get('id'), 0),
                    'league_name': str(league.get('name', '')),
                })
        return pd.DataFrame(rows), None
    except Exception as exc:
        return pd.DataFrame(), str(exc)


@st.cache_data(ttl=120)
def cached_live(game_pk: int | None):
    return get_live_summary(game_pk)


@st.cache_data(ttl=1800)
def cached_statcast(team_abbr: str, start_date: str, end_date: str, player_type: str):
    return get_statcast_team_df(team_abbr=team_abbr, start_date=start_date, end_date=end_date, player_type=player_type)


@st.cache_data(ttl=1800)
def cached_linescore(game_pk: int):
    try:
        response = requests.get(f'{BASE_URL}/game/{game_pk}/linescore', timeout=TIMEOUT, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        innings = []
        for inn in data.get('innings', []):
            innings.append({
                'inning': coerce_int(inn.get('num'), 0),
                'home_runs': coerce_int((inn.get('home') or {}).get('runs'), 0),
                'away_runs': coerce_int((inn.get('away') or {}).get('runs'), 0),
            })
        return {'innings': innings}, None
    except Exception as exc:
        return {}, str(exc)


def _fmt_pct(value: float, digits: int = 1) -> str:
    return f'{coerce_float(value, 0.0):.{digits}f}%'


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce')
    return out


def build_last_three_df(recent_games_df: pd.DataFrame) -> pd.DataFrame:
    cols = ['Date', 'Opponent', 'Home/Away', 'Final', 'Result', 'Runs', 'Allowed']
    if recent_games_df.empty:
        return pd.DataFrame(columns=cols)
    out = recent_games_df.tail(3).copy()
    out['Final'] = out['Team Runs'].astype(str) + '-' + out['Opp Runs'].astype(str)
    out['Home/Away'] = out['Location']
    out['Runs'] = out['Team Runs']
    out['Allowed'] = out['Opp Runs']
    return out[cols].reset_index(drop=True)


def build_future_three_df(full_schedule_df: pd.DataFrame, team_id: int, recent_games_df: pd.DataFrame) -> pd.DataFrame:
    cols = ['Date', 'Opponent', 'Home/Away', 'Win %', 'Confidence', 'Key Factor']
    if full_schedule_df.empty:
        return pd.DataFrame(columns=cols)

    future_df = full_schedule_df[
        ~full_schedule_df['status'].isin(['Final', 'Game Over', 'Completed Early'])
    ].copy()

    if future_df.empty:
        return pd.DataFrame(columns=cols)

    future_df = future_df.sort_values(['officialDate', 'gameDate']).head(3).copy()
    recent_diff = coerce_float(recent_games_df.tail(min(5, len(recent_games_df)))['Run Diff'].mean(), 0.0) if not recent_games_df.empty else 0.0

    rows = []
    for _, row in future_df.iterrows():
        is_home = coerce_int(row.get('home_id'), 0) == team_id
        opponent = row.get('away') if is_home else row.get('home')
        home_away = 'Home' if is_home else 'Away'
        home_edge = 0.04 if is_home else -0.04
        form_edge = max(min(recent_diff / 10.0, 0.10), -0.10)
        base = 0.50 + home_edge + form_edge
        win_prob = max(min(base, 0.75), 0.25)

        if win_prob >= 0.58:
            confidence = 'High'
            key = 'Recent run differential edge'
        elif win_prob >= 0.52:
            confidence = 'Medium'
            key = 'Slight form edge'
        elif win_prob <= 0.42:
            confidence = 'High'
            key = 'Current form risk'
        else:
            confidence = 'Low'
            key = 'Tight matchup'

        rows.append({
            'Date': row.get('officialDate', row.get('gameDate', '')),
            'Opponent': opponent,
            'Home/Away': home_away,
            'Win %': _fmt_pct(win_prob * 100, 1),
            'Confidence': confidence,
            'Key Factor': key,
        })

    return pd.DataFrame(rows, columns=cols)


def build_playoff_tracker(standings_row: 'pd.Series | None', recent_games_df: pd.DataFrame) -> pd.DataFrame:
    columns = ['Metric', 'Value', 'Trend']
    if standings_row is None or standings_row.empty:
        return pd.DataFrame(columns=columns)

    win_pct = coerce_float(standings_row.get('win_pct'), 0.0)
    wc_rank = coerce_int(standings_row.get('wild_card_rank'), 99)
    division_rank = coerce_int(standings_row.get('division_rank'), 99)
    games_back = coerce_float(standings_row.get('games_back'), 99.0)
    wc_games_back = coerce_float(standings_row.get('wc_games_back'), 99.0)
    recent_diff = coerce_float(recent_games_df.tail(min(10, len(recent_games_df)))['Run Diff'].mean(), 0.0) if not recent_games_df.empty else 0.0

    estimated = (
        (win_pct * 100.0) * 0.58
        + max(0.0, (7 - min(wc_rank, 7)) / 6) * 22
        + max(0.0, (6 - min(division_rank, 6)) / 5) * 10
        + max(0.0, 10.0 - min(wc_games_back, 10.0)) * 0.7
        + max(min(recent_diff, 5.0), -5.0) * 2.0
    )
    estimated = max(min(estimated, 99.0), 1.0)

    # Display-friendly rank values (blank if not ranked)
    div_rank_display = str(division_rank) if division_rank < 99 else '-'
    wc_rank_display = str(wc_rank) if wc_rank < 99 else '-'
    gb_display = games_back if games_back < 99.0 else '-'
    wcgb_display = wc_games_back if wc_games_back < 99.0 else '-'

    rows = [
        {'Metric': 'Estimated Playoff %', 'Value': _fmt_pct(estimated, 1), 'Trend': stoplight(recent_diff, neutral_band=0.2)},
        {'Metric': 'Division Rank', 'Value': div_rank_display, 'Trend': stoplight(3 - division_rank, neutral_band=0.5)},
        {'Metric': 'Wild Card Rank', 'Value': wc_rank_display, 'Trend': stoplight(4 - wc_rank, neutral_band=0.5)},
        {'Metric': 'Games Back', 'Value': gb_display, 'Trend': stoplight(2 - games_back, neutral_band=0.3)},
        {'Metric': 'WC Games Back', 'Value': wcgb_display, 'Trend': stoplight(2 - wc_games_back, neutral_band=0.3)},
    ]
    return pd.DataFrame(rows, columns=columns)


def build_inning_impact_df(season_df: pd.DataFrame, team_id: int) -> pd.DataFrame:
    columns = ['Inning', 'Runs For', 'Runs Against', 'Net Impact', 'Stoplight']
    if season_df.empty:
        return pd.DataFrame(columns=columns)

    completed = season_df[season_df['status'].isin(['Final', 'Game Over', 'Completed Early'])].copy()
    if completed.empty:
        return pd.DataFrame(columns=columns)

    totals: dict[int, dict[str, int]] = {}
    for _, row in completed.iterrows():
        game_pk = coerce_int(row.get('gamePk'), 0)
        if game_pk <= 0:
            continue
        linescore, error = cached_linescore(game_pk)
        if error or not linescore:
            continue
        innings = linescore.get('innings', [])
        is_home = coerce_int(row.get('home_id'), 0) == team_id
        for inn in innings:
            inning_num = coerce_int(inn.get('inning'), 0)
            if inning_num <= 0:
                continue
            rf = coerce_int(inn.get('home_runs') if is_home else inn.get('away_runs'), 0)
            ra = coerce_int(inn.get('away_runs') if is_home else inn.get('home_runs'), 0)
            totals.setdefault(inning_num, {'Runs For': 0, 'Runs Against': 0})
            totals[inning_num]['Runs For'] += rf
            totals[inning_num]['Runs Against'] += ra

    rows = []
    for inning in sorted(totals):
        rf = totals[inning]['Runs For']
        ra = totals[inning]['Runs Against']
        net = rf - ra
        rows.append({
            'Inning': inning,
            'Runs For': rf,
            'Runs Against': ra,
            'Net Impact': net,
            'Stoplight': '🟢 Green' if net > 0 else '🟡 Yellow' if net == 0 else '🔴 Red',
        })

    return pd.DataFrame(rows, columns=columns)


def build_hr_distance_split(statcast_batter_df: pd.DataFrame, team_abbr: str) -> pd.DataFrame:
    cols = ['Split', 'Avg HR Distance']
    if statcast_batter_df.empty:
        return pd.DataFrame(columns=cols)

    df = _to_numeric(statcast_batter_df.copy(), ['hit_distance_sc'])
    if 'events' not in df.columns or 'hit_distance_sc' not in df.columns:
        return pd.DataFrame(columns=cols)

    hr_df = df[df['events'].astype(str) == 'home_run'].copy()
    if hr_df.empty:
        return pd.DataFrame(columns=cols)

    if 'home_team' in hr_df.columns:
        hr_df['Split'] = hr_df['home_team'].astype(str).str.upper().eq(team_abbr.upper()).map({True: 'Home', False: 'Away'})
    elif 'inning_topbot' in hr_df.columns:
        hr_df['Split'] = hr_df['inning_topbot'].map(lambda x: 'Away' if str(x).lower() == 'top' else 'Home')
    else:
        hr_df['Split'] = 'All'

    out = hr_df.groupby('Split', as_index=False)['hit_distance_sc'].mean().rename(columns={'hit_distance_sc': 'Avg HR Distance'})
    out['Avg HR Distance'] = out['Avg HR Distance'].round(1)
    return out[cols]


def build_hitter_ev_df(statcast_batter_df: pd.DataFrame) -> pd.DataFrame:
    cols = ['Hitter', 'BIP', 'Avg Exit Velo']
    if statcast_batter_df.empty:
        return pd.DataFrame(columns=cols)

    df = _to_numeric(statcast_batter_df.copy(), ['launch_speed'])
    if 'launch_speed' not in df.columns:
        return pd.DataFrame(columns=cols)

    name_col = 'player_name' if 'player_name' in df.columns else 'batter_name' if 'batter_name' in df.columns else None
    if not name_col:
        return pd.DataFrame(columns=cols)

    contact = df[df['launch_speed'] > 0].copy()
    if contact.empty:
        return pd.DataFrame(columns=cols)

    out = contact.groupby(name_col, as_index=False).agg(BIP=('launch_speed', 'size'), **{'Avg Exit Velo': ('launch_speed', 'mean')})
    out = out.rename(columns={name_col: 'Hitter'})
    out['Avg Exit Velo'] = out['Avg Exit Velo'].round(1)
    return out.sort_values(['Avg Exit Velo', 'BIP'], ascending=[False, False]).head(12).reset_index(drop=True)


def build_pitcher_ev_allowed_df(statcast_pitcher_df: pd.DataFrame) -> pd.DataFrame:
    cols = ['Pitcher', 'BIP Against', 'Avg EV Allowed']
    if statcast_pitcher_df.empty:
        return pd.DataFrame(columns=cols)

    df = _to_numeric(statcast_pitcher_df.copy(), ['launch_speed'])
    if 'launch_speed' not in df.columns:
        return pd.DataFrame(columns=cols)

    name_col = 'player_name' if 'player_name' in df.columns else 'pitcher_name' if 'pitcher_name' in df.columns else None
    if not name_col:
        return pd.DataFrame(columns=cols)

    contact = df[df['launch_speed'] > 0].copy()
    if contact.empty:
        return pd.DataFrame(columns=cols)

    out = contact.groupby(name_col, as_index=False).agg(**{'BIP Against': ('launch_speed', 'size'), 'Avg EV Allowed': ('launch_speed', 'mean')})
    out = out.rename(columns={name_col: 'Pitcher'})
    out['Avg EV Allowed'] = out['Avg EV Allowed'].round(1)
    return out.sort_values(['Avg EV Allowed', 'BIP Against'], ascending=[True, False]).head(12).reset_index(drop=True)


# ── App layout ──────────────────────────────────────────────────────────────

st.title('Live MLB Analytics Dashboard')
st.caption('Streamlit build with standings, playoff tracker, last-3 scoreboard, future-3 forecast, inning stoplights, and Statcast views.')

with st.container(border=True):
    st.markdown(
        'Live MLB + Statcast integration with last-3 scoreboard, next-3 preview, standings, playoff tracker, and season inning impact stoplights.'
    )

teams_df, teams_warning = cached_teams()
if teams_warning:
    st.warning(teams_warning)

if teams_df.empty:
    st.error('Teams could not be loaded. The app shell is running, but no team list is available.')
    st.stop()

team_names = teams_df['name'].tolist()
default_team = 'Kansas City Royals' if 'Kansas City Royals' in team_names else team_names[0]

with st.sidebar:
    st.header('Controls')
    selected_team = st.selectbox('Select team', team_names, index=team_names.index(default_team))
    selected_date = st.date_input('Schedule date', value=date.today())
    statcast_window = st.slider('Statcast lookback days', min_value=7, max_value=60, value=21, step=7)
    if st.button('Refresh Data', use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption('Advanced sections use optional Statcast data. If unavailable, the rest of the app still loads.')

team_row = safe_team_row(teams_df, selected_team)
team_id = coerce_int(team_row.get('id') if team_row is not None else 0, 0)
team_abbr = str(team_row.get('abbreviation') if team_row is not None else '').upper()
season = selected_date.year
selected_date_str = str(selected_date)
statcast_start = str(selected_date - timedelta(days=statcast_window))

daily_df, daily_error = cached_daily(team_id, selected_date_str)
season_df, season_error = cached_season(team_id, season, selected_date_str)
full_schedule_df, full_schedule_error = cached_full_schedule(team_id, season)
standings_df, standings_error = cached_standings(season)

for error_msg, label in [
    (daily_error, 'Daily schedule'),
    (season_error, 'Season schedule'),
    (full_schedule_error, 'Full schedule'),
    (standings_error, 'Standings'),
]:
    if error_msg:
        st.warning(f'{label} call had an issue: {error_msg}')

snapshot = build_team_snapshot(team_row, season_df, daily_df)
summary_df = build_summary_df(snapshot)
trend_df = build_trend_df(season_df, selected_team)
recent_games_df = build_recent_games_df(season_df, selected_team, count=10)
rolling_df = build_team_rolling_df(recent_games_df)
schedule_table = build_schedule_table(daily_df, selected_team)
kpi_cards = build_kpi_cards(snapshot, trend_df)

last_three_df = build_last_three_df(recent_games_df)
future_three_df = build_future_three_df(full_schedule_df, team_id, recent_games_df)
standings_match = standings_df[standings_df['team_id'] == team_id] if not standings_df.empty and 'team_id' in standings_df.columns else pd.DataFrame()
selected_standings_row = standings_match.iloc[0] if not standings_match.empty else None
playoff_df = build_playoff_tracker(selected_standings_row, recent_games_df)
division_view_df, wildcard_view_df = build_standings_views(standings_df, selected_standings_row)
inning_impact_df = build_inning_impact_df(season_df, team_id)

# Live feed — use daily schedule which has abstract_status for better detection
live_game_pk = choose_live_game_pk(daily_df)
live_summary, live_error = cached_live(live_game_pk)
if live_game_pk and live_error:
    st.caption('Live game feed is not available from MLB right now. Other tabs remain active.')

statcast_batter_df, statcast_batter_error = cached_statcast(team_abbr, statcast_start, selected_date_str, 'batter')
statcast_pitcher_df, statcast_pitcher_error = cached_statcast(team_abbr, statcast_start, selected_date_str, 'pitcher')

batter_grades_df = build_batter_grades_df(statcast_batter_df)
pitcher_grades_df = build_pitcher_grades_df(statcast_pitcher_df)
pitch_mix_df = build_pitch_mix_df(statcast_pitcher_df)
statcast_summary_df = build_statcast_summary_df(statcast_batter_df, statcast_pitcher_df)
hitter_ev_df = build_hitter_ev_df(statcast_batter_df)
pitcher_ev_allowed_df = build_pitcher_ev_allowed_df(statcast_pitcher_df)
hr_distance_df = build_hr_distance_split(statcast_batter_df, team_abbr)

extra_card = {
    'label': 'Playoff %',
    'value': playoff_df.iloc[0]['Value'] if not playoff_df.empty else 'N/A',
    'delta': playoff_df.iloc[0]['Trend'] if not playoff_df.empty else '🟡 Even',
}
cols = st.columns(5)
for col, item in zip(cols, kpi_cards + [extra_card]):
    col.metric(item['label'], item['value'], item['delta'])

summary_tab, schedule_tab, trends_tab, deep_tab, live_tab = st.tabs(['Summary', 'Schedule', 'Trends', 'Deep Trends', 'Live Feed'])

with summary_tab:
    st.subheader('Team Summary')
    st.caption('Current team snapshot, last 3 completed games, playoff tracker, and standings context.')
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('#### Last 3 Games Scoreboard')
        st.caption('Date, opponent, home/away, final score, result, runs scored, and runs allowed.')
        if last_three_df.empty:
            st.info('No completed games are available yet.')
        else:
            st.dataframe(last_three_df, use_container_width=True, hide_index=True)
    with c2:
        st.markdown('#### Next 3 Games Preview')
        st.caption('Upcoming schedule with estimated win probability, confidence, and key factor.')
        if future_three_df.empty:
            st.info('No upcoming games are available yet.')
        else:
            st.dataframe(future_three_df, use_container_width=True, hide_index=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('#### Playoff % Tracker')
        st.caption('Estimated playoff probability plus rank and games-back context.')
        if playoff_df.empty:
            st.info('Playoff tracker is not available yet.')
        else:
            st.dataframe(playoff_df, use_container_width=True, hide_index=True)
    with c4:
        st.markdown('#### Trend Indicators')
        st.caption('Stoplight trends for scoring, prevention, record, and consistency.')
        if trend_df.empty:
            st.info('Trend data is not available yet.')
        else:
            st.dataframe(trend_df, use_container_width=True, hide_index=True)

    c5, c6 = st.columns(2)
    with c5:
        st.markdown('#### Division Standings')
        if division_view_df.empty:
            st.info('Division standings are not available yet.')
        else:
            st.dataframe(division_view_df, use_container_width=True, hide_index=True)
    with c6:
        st.markdown('#### Wild Card Standings')
        if wildcard_view_df.empty:
            st.info('Wild card standings are not available yet.')
        else:
            st.dataframe(wildcard_view_df, use_container_width=True, hide_index=True)

with schedule_tab:
    st.subheader('Selected Date Schedule')
    st.caption('Games for the selected date plus the next 3 game look-ahead view.')
    if schedule_table.empty:
        st.info('No games found for the selected team and date.')
    else:
        st.dataframe(schedule_table, use_container_width=True, hide_index=True)
        render_schedule_chart(daily_df)

    st.markdown('#### Future 3 Games')
    if future_three_df.empty:
        st.info('No upcoming games are available yet.')
    else:
        st.dataframe(future_three_df, use_container_width=True, hide_index=True)

with trends_tab:
    st.subheader('Trends')
    st.caption('Rolling 3 game averages, run charts, inning heat maps, and full-season inning impact stoplights.')
    render_recent_trend_chart(recent_games_df)
    render_run_diff_chart(recent_games_df)

    st.markdown('#### Rolling 3 Game Averages')
    if rolling_df.empty:
        st.info('Rolling trend data is not available yet.')
    else:
        st.dataframe(rolling_df, use_container_width=True, hide_index=True)
        render_rolling_chart(rolling_df)

    st.markdown('#### Season Inning Impact Stoplight View')
    st.caption('Green if net impact is positive, yellow if neutral, red if negative across the full season.')
    if inning_impact_df.empty:
        st.info('Inning impact is not available yet.')
    else:
        st.dataframe(inning_impact_df, use_container_width=True, hide_index=True)

        heat_df = inning_impact_df.copy()
        fig_for = px.imshow(
            [heat_df['Runs For'].tolist()],
            x=heat_df['Inning'].astype(str).tolist(),
            y=['Runs For'],
            aspect='auto',
            color_continuous_scale='Greens',
            text_auto=True,
            title='Runs For by Inning',
        )
        fig_for.update_layout(
            height=240,
            margin=dict(l=20, r=20, t=50, b=20),
            dragmode=False,
            xaxis=dict(fixedrange=True),
            yaxis=dict(fixedrange=True),
        )
        st.plotly_chart(fig_for, use_container_width=True, config=_NO_ZOOM_CONFIG)

        fig_against = px.imshow(
            [heat_df['Runs Against'].tolist()],
            x=heat_df['Inning'].astype(str).tolist(),
            y=['Runs Against'],
            aspect='auto',
            color_continuous_scale='Reds',
            text_auto=True,
            title='Runs Against by Inning',
        )
        fig_against.update_layout(
            height=240,
            margin=dict(l=20, r=20, t=50, b=20),
            dragmode=False,
            xaxis=dict(fixedrange=True),
            yaxis=dict(fixedrange=True),
        )
        st.plotly_chart(fig_against, use_container_width=True, config=_NO_ZOOM_CONFIG)

with deep_tab:
    st.subheader('Deep Trends')
    st.caption(f'Statcast window: {statcast_start} through {selected_date_str}. These sections remain optional and fail gracefully if Statcast is unavailable.')

    if statcast_batter_error and batter_grades_df.empty:
        st.info(f'Batter Statcast data is unavailable right now: {statcast_batter_error}')
    if statcast_pitcher_error and pitcher_grades_df.empty and pitch_mix_df.empty:
        st.info(f'Pitcher Statcast data is unavailable right now: {statcast_pitcher_error}')

    t1, t2 = st.columns(2)
    with t1:
        st.markdown('#### Team Statcast Snapshot')
        st.dataframe(statcast_summary_df, use_container_width=True, hide_index=True)
    with t2:
        st.markdown('#### Pitch Type Percentage and Spin Rate')
        if pitch_mix_df.empty:
            st.info('Pitch mix is not available yet.')
        else:
            st.dataframe(pitch_mix_df, use_container_width=True, hide_index=True)
            render_pitch_mix_chart(pitch_mix_df)

    t3, t4 = st.columns(2)
    with t3:
        st.markdown('#### Hitter Avg Exit Velocity')
        if hitter_ev_df.empty:
            st.info('Hitter exit velocity is not available yet.')
        else:
            st.dataframe(hitter_ev_df, use_container_width=True, hide_index=True)
            if not batter_grades_df.empty:
                render_statcast_scatter(batter_grades_df)
    with t4:
        st.markdown('#### Pitcher Avg Exit Velocity Allowed')
        if pitcher_ev_allowed_df.empty:
            st.info('Pitcher exit velocity allowed is not available yet.')
        else:
            st.dataframe(pitcher_ev_allowed_df, use_container_width=True, hide_index=True)

    t5, t6 = st.columns(2)
    with t5:
        st.markdown('#### Player Impact Table')
        st.caption('Batter grades: exit velocity, hard hit %, xwOBA, XBH rate, and whiff avoidance.')
        if batter_grades_df.empty:
            st.info('Player impact data is not available yet.')
        else:
            st.dataframe(batter_grades_df, use_container_width=True, hide_index=True)
    with t6:
        st.markdown('#### Pitching Table')
        st.caption('Pitcher grades: whiff rate, wOBA allowed, velocity, spin rate, and strike efficiency.')
        if pitcher_grades_df.empty:
            st.info('Pitching data is not available yet.')
        else:
            st.dataframe(pitcher_grades_df, use_container_width=True, hide_index=True)

    st.markdown('#### Avg Home Run Distance, Home vs Away')
    if hr_distance_df.empty:
        st.info('Home/away home run distance split is not available yet.')
    else:
        st.dataframe(hr_distance_df, use_container_width=True, hide_index=True)

with live_tab:
    st.subheader('Live Feed')
    st.caption('In-progress game box score. Refreshes every 2 minutes while a game is active.')
    live_df = build_live_box_df(live_summary)
    if live_df.empty:
        st.info('No active in-progress live game feed is available for the selected team and date.')
    else:
        st.dataframe(live_df, use_container_width=True, hide_index=True)

st.divider()
st.caption('Last 3 games scoreboard · next 3 preview · standings · estimated playoff tracker · inning stoplights · heat maps · Statcast tables.')
