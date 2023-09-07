"""
This file contains all the code related to the Clash Royale's API.
The functions are for retrieving the most up-to-date game statistics.
"""
from typing import Optional, List
import requests
import json

with open('../config.json') as f:
    config = json.load(f)

API_TOKEN = config['api_token']


def fetch_player(player_tag: str) -> Optional[dict]:
    """ Fetch player data from the Clash Royale API
    :param player_tag: the player tag
    :return: a dict containing the player data
    """
    url = f"https://api.clashroyale.com/v1/players/%23{player_tag}"

    headers = {
        "Authorization": f"Bearer {API_TOKEN}"
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f'failed with status code: {response.status_code}')
        return None


def get_player_name(player_tag: str) -> Optional[str]:
    """ Get the player name from the player tag
    :param player_tag: the player tag
    :return: the player name
    """
    response = fetch_player(player_tag)
    if response is not None:
        return response["name"]
    else:
        return None


def get_player_pb(player_tag: str) -> Optional[int]:
    """ Get the player personal best from the player tag
    :param player_tag: the player tag
    :return: the player personal best
    """
    response = fetch_player(player_tag)
    if response is not None:
        return response["bestTrophies"]
    else:
        return None


def get_player_best_rank(player_tag: str) -> Optional[int]:
    """ Get the player best rank from the player tag
    :param player_tag: the player tag
    :return: the player best rank
    """
    response = fetch_player(player_tag)
    if response is not None:
        return response["leagueStatistics"]["bestSeason"]["rank"] \
            if 'rank' in response["leagueStatistics"]["bestSeason"].keys() else 0
    else:
        return None


def get_battle_logs(player_tag: str) -> Optional[List[dict]]:
    """ Get the player battle logs from the player tag
    :param player_tag: the player tag
    :return: a list of dicts containing the player battle logs
    """
    url = f"https://api.clashroyale.com/v1/players/%23{player_tag}/battlelog"
    headers = {
        "Authorization": f"Bearer {API_TOKEN}"
    }
    if requests.get(url, headers=headers).status_code == 200:
        return requests.get(url, headers=headers).json()
    else:
        print(requests.get(url, headers=headers).status_code)


def get_league_seasons() -> Optional[List[str]]:
    """ Get the league seasons
    :return: a list of dicts containing the league seasons
    """
    url = "https://api.clashroyale.com/v1/locations/global/seasons"
    headers = {
        "Authorization": f"Bearer {API_TOKEN}"
    }
    if requests.get(url, headers=headers).status_code == 200:
        return requests.get(url, headers=headers).json()["items"]
    else:
        print(requests.get(url, headers=headers).status_code)


def get_season_ranking(identifier: str, limit: int) -> Optional[List[str]]:
    """ Get the season ranking
    :param identifier: the season identifier
    :param limit: the limit on the amount of ranks returned
    :return: a list of dicts containing the season ranking
    """
    url = f"https://api.clashroyale.com/v1/locations/global/pathoflegend/{identifier}/rankings/players?limit={limit}"
    headers = {
        "Authorization": f"Bearer {API_TOKEN}"
    }
    if requests.get(url, headers=headers).status_code == 200:
        return requests.get(url, headers=headers).json()
    else:
        print(requests.get(url, headers=headers).status_code)


def get_all_cards() -> Optional[List[str]]:
    """ Get all the cards
    :return: a list containing all the cards
    """
    url = "https://api.clashroyale.com/v1/cards"
    headers = {
        "Authorization": f"Bearer {API_TOKEN}"
    }
    result = requests.get(url, headers=headers).json()
    if result:
        return result["items"]
    else:
        print(requests.get(url, headers=headers).status_code)


def check_gt_rank(player_tag: str) -> bool:
    """ Check if the player has ever finished top 1000
    in the global tournament, return True if yes
    """
    response = fetch_player(player_tag)
    if response is not None:
        return any(x for x in response['badges'] if x['name'] == 'LadderTournamentTop1000')
    else:
        return False


def get_amount_gt(player_tag: str) -> int:
    """ Return the amount of time a player has finished top 1000 in GT"""
    response = fetch_player(player_tag)
    if response is not None:
        gt = [x for x in response['badges'] if x['name'] == 'LadderTournamentTop1000']
        if len(gt) == 0:
            return 0
        else:
            return gt[0]['level']


def get_max_wins(player_tag: str) -> int:
    """ Return the maximum amount of wins the players got in a challenge"""
    response = fetch_player(player_tag)
    if response is not None:
        return response['challengeMaxWins']
    else:
        return 0


def get_cards_won(player_tag: str) -> int:
    """ Return the maximum amount of cards a player has won"""
    response = fetch_player(player_tag)
    if response is not None:
        return response['challengeCardsWon']
    else:
        return 0


def get_pol_best_rank(player_tag: str) -> int:
    """ Return the best rank achieved in pathOfLegends"""
    response = fetch_player(player_tag)
    if response is not None:
        rank = response['bestPathOfLegendSeasonResult']['rank']
        return rank if rank else 0
    else:
        return 0

