import pandas as pd
from collections import defaultdict
from collections import deque
import numpy as np


'''
Goal: Create a two new columns that tracks the win streak of each team at every game

maintain a map of all the teams and update += 1 if team has won else set back to 0

- Find the teams playing
- Increment the winner team score
'''


def create_streaks(nba_data):
    streak_map = defaultdict(int)
    for index, row in nba_data.iterrows():
        # Label rows
        home, away, winner = row['Home Team'], row['Away Team'], row['Winner']

        # Update team streak
        nba_data.at[index, "Home Team Streak"] = streak_map[home]
        nba_data.at[index, "Away Team Streak"] = streak_map[away]

        # Calculate streak
        streak_map[home] += 1
        streak_map[home if home != winner else away] = 0
    return nba_data


def rolling_avgs(nba_data, param_home, param_away):
    team_details = defaultdict(deque)
    col_home = param_home + " Rolling Avg"
    col_away = param_away + " Rolling Avg"

    for index, row in nba_data.iterrows():
        home, away = row['Home Team'], row['Away Team']
        if len(team_details[home]) == 5:
            nba_data.at[index, col_home] = sum(
                team_details[home])/len(team_details[home])
            team_details[home].popleft()
        else:
            nba_data.at[index, col_home] = row[param_home]

        if len(team_details[away]) == 5:
            nba_data.at[index, col_away] = sum(
                team_details[away])/len(team_details[away])
            team_details[away].popleft()
        else:
            nba_data.at[index, col_away] = row[param_away]

        # if home == 'BOS' and len(team_details['BOS']) > 0:
        #     print(team_details['BOS'], row['reb_home'], sum(
        #         team_details[home])/len(team_details[home]), row['game_date'])
        # if away == 'BOS' and len(team_details['BOS']) > 0:
        #     print(team_details['BOS'], row['reb_away'], sum(
        #         team_details[away])/len(team_details[away]), row['game_date'])

        team_details[home].append(row[param_home])
        team_details[away].append(row[param_away])
    return nba_data


def create_rolling_avgs(nba_data):
    columns_to_average_home = [
        'fg_pct_home', 'fg3_pct_home', 'ft_pct_home', 'reb_home', 'ast_home',
        'stl_home', 'blk_home', 'tov_home', 'plus_minus_home']
    columns_to_average_away = [
        'fg_pct_away', 'fg3_pct_away', 'ft_pct_away', 'reb_away', 'ast_away',
        'stl_away', 'blk_away', 'tov_away', 'plus_minus_away'
    ]

    for column_home, column_away in zip(columns_to_average_home, columns_to_average_away):
        rolling_avgs(
            nba_data, column_home, column_away)

    return nba_data


def calc_records(nba_data):
    teamRecord = defaultdict(int)
    totalGames = defaultdict(int)

    for index, row in nba_data.iterrows():
        home_team = row['Home Team']
        away_team = row['Away Team']
        winner = row['Winner']

        # record_home updated
        if totalGames[home_team] == 0:
            # for first game of season, both records are 0-0
            nba_data.at[index, 'record_home'] = 0.0
        elif teamRecord[home_team] == totalGames[home_team] - 1:
            # for undefeated teams (such as 1-0 or 2-0). cant divide by zero so manually set?
            nba_data.at[index, 'record_home'] = 1.0
        else:
            nba_data.at[index, 'record_home'] = teamRecord[home_team] / \
                totalGames[home_team]

        # record_away updated
        if totalGames[away_team] == 0:
            # for first game of season, both records are 0-0
            nba_data.at[index, 'record_away'] = 0.0
        elif teamRecord[away_team] == totalGames[away_team] - 1:
            # for undefeated teams (such as 1-0 or 2-0). cant divide by zero so manually set?
            nba_data.at[index, 'record_away'] = 1.0
        else:
            nba_data.at[index, 'record_away'] = teamRecord[away_team] / \
                totalGames[away_team]

        totalGames[home_team] += 1
        totalGames[away_team] += 1

        if winner == home_team:
            teamRecord[home_team] += 1
        elif winner == away_team:
            teamRecord[away_team] += 1

    return nba_data


def prep_data(raw):
    df = pd.read_csv(raw)
    df.replace(' ', pd.NA, inplace=True)
    numeric_columns = df.select_dtypes(include='number').columns
    df[numeric_columns] = df[numeric_columns].fillna(
        df[numeric_columns].mean())

    # Convert the 'dates' column to datetime format
    df['game_date'] = pd.to_datetime(df['game_date'])

    # Sort the DataFrame by the 'dates' column
    df.sort_values(by='game_date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Filter the dataframe to keep only the regular season games from dates between October 25, 2016 and June 12, 2023
    df = df[(df['game_date'] >= '2016-10-25') & (df['game_date'] <= '2023-06-12') & (df['season_type'] == 'Regular Season')]

    # Clean dataset by dropping any unecessary columns
    columns_to_drop = ['season_id', 'team_id_home', 'team_name_home', 'game_id', 'matchup_home', 'min', 'fgm_home', 'fga_home', 'fg3m_home', 'fg3a_home', 'ftm_home', 'fta_home', 'oreb_home', 'dreb_home', 'pf_home', 'pts_home', 'video_available_home',
                       'team_id_away', 'team_name_away', 'matchup_away', 'wl_away', 'fgm_away', 'fga_away', 'fg3m_away', 'fg3a_away', 'ftm_away', 'fta_away', 'oreb_away', 'dreb_away', 'pf_away', 'pts_away', 'video_available_away', 'pts_away', 'pf_away',
                       'season_type', ]
    df.drop(columns=columns_to_drop, inplace=True)

    # Restructure dataset
    # Extract data from 'team_abbreviation_away' column
    awayTeam = df['team_abbreviation_away']

    # Drop the 'team_abbreviation_away' column temporarily
    df.drop(columns=['team_abbreviation_away'], inplace=True)

    # Insert 'team_abbreviation_away' column to the location where 'game_date' was
    df.insert(df.columns.get_loc('game_date'),
              'team_abbreviation_away', awayTeam)

    # Duplicate 'wl_home' column
    df.insert(df.columns.get_loc('wl_home'), 'wl_home2', df['wl_home'])

    # Name columns
    df.columns = ['Home Team', 'Away Team', 'Game Date', 'Winner', 'Home Win/Loss', 'fg_pct_home',
                  'fg3_pct_home', 'ft_pct_home', 'reb_home', 'ast_home', 'stl_home',
                  'blk_home', 'tov_home', 'plus_minus_home', 'fg_pct_away', 'fg3_pct_away',
                  'ft_pct_away', 'reb_away', 'ast_away', 'stl_away', 'blk_away', 'tov_away', 'plus_minus_away']

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        if row['Winner'] == 'W':
            # If the home team won, set the winner abbreviation to the home team abbreviation
            df.at[index, 'Winner'] = row['Home Team']
        else:
            # If the away team won, set the winner abbreviation to the away team abbreviation
            df.at[index, 'Winner'] = row['Away Team']

    # Display the filtered DataFrame
    # df.to_csv("clean_data.csv", index=False)
    return df


def remove_columns(nba_data):
    return nba_data.drop(columns=['Home Team', 'Away Team', 'Game Date', 'Winner', 'reb_home', 'reb_away', 'fg_pct_home', 'fg3_pct_home', 'ft_pct_home', 'ast_home',
                                  'stl_home', 'blk_home', 'tov_home', 'plus_minus_home',
                                  'fg_pct_away', 'fg3_pct_away', 'ft_pct_away', 'ast_away',
                                  'stl_away', 'blk_away', 'tov_away', 'plus_minus_away'])


if __name__ == '__main__':
    nba_data = prep_data('../data/game.csv')
    nba_data = create_streaks(nba_data)
    nba_data = calc_records(nba_data)
    nba_data = create_rolling_avgs(nba_data)
    final_data = remove_columns(nba_data)
    print(final_data.head())
    final_data.to_csv("../data/final_data.csv", index=False)
