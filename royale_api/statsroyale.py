from typing import Optional, List

import requests

API_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiIsImtpZCI6IjI4YTMxOGY3LTAwMDAtYTFlYi03ZmExL" \
            "TJjNzQzM2M2Y2NhNSJ9.eyJpc3MiOiJzdXBlcmNlbGwiLCJhdWQiOiJzdXBlcmNlbGw6Z2FtZWFwa" \
            "SIsImp0aSI6IjdlM2JiNTU1LTE5NDItNDYwMi05ZDc5LWY5YjY4ZGYxYmFkYSIsImlhdCI6MTY4Mz" \
            "I0ODYyMywic3ViIjoiZGV2ZWxvcGVyL2UwZmM5ZjgxLWQzZDgtZGI3OS1lYzY1LTNiZDg4MTRlZmN" \
            "iMCIsInNjb3BlcyI6WyJyb3lhbGUiXSwibGltaXRzIjpbeyJ0aWVyIjoiZGV2ZWxvcGVyL3NpbHZl" \
            "ciIsInR5cGUiOiJ0aHJvdHRsaW5nIn0seyJjaWRycyI6WyIxNTguMTgxLjEyMy4xNiJdLCJ0eXBlI" \
            "joiY2xpZW50In1dfQ.bRVwGGSEmS5PzWjFPXbuDSGaAuyQtYl1noZ5yhOhQC5SXZuEu6HwW32Nz8" \
            "brR2JbR9GOkaGztjyD4ZKBAeQtpw"


def fetch_player(player_tag: str) -> Optional[dict]:
    url = f"https://api.clashroyale.com/v1/players/%23{player_tag}"

    headers = {
        "Authorization": f"Bearer {API_TOKEN}"
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        # print the error message
        print(response.json()["message"])
        # print error code
        print(response.status_code)
        return None


def get_player_name(player_tag: str) -> str:
    response = fetch_player(player_tag)
    if response is not None:
        return response["name"]


def get_player_pb(player_tag: str) -> int:
    response = fetch_player(player_tag)
    if response is not None:
        return response["bestTrophies"]


def get_player_best_rank(player_tag: str) -> int:
    response = fetch_player(player_tag)
    if response is not None:
        return response["leagueStatistics"]["bestSeason"]["rank"]


def get_battle_logs(player_tag: str) -> Optional[List[dict]]:
    url = f"https://api.clashroyale.com/v1/players/%23{player_tag}/battlelog"
    headers = {
        "Authorization": f"Bearer {API_TOKEN}"
    }
    if requests.get(url, headers=headers).status_code == 200:
        return requests.get(url, headers=headers).json()
    else:
        return None


def get_league_seasons() -> List[str]:
    url = "https://api.clashroyale.com/v1/locations/global/seasons"
    headers = {
        "Authorization": f"Bearer {API_TOKEN}"
    }
    return requests.get(url, headers=headers).json()


def get_season_ranking(identifier: str) -> List[str]:
    url = f"https://api.clashroyale.com/v1/locations/global/seasons/{identifier}/rankings/players"
    headers = {
        "Authorization": f"Bearer {API_TOKEN}"
    }
    return requests.get(url, headers=headers).json()


def get_all_cards():
    url = "https://api.clashroyale.com/v1/cards"
    headers = {
        "Authorization": f"Bearer {API_TOKEN}"
    }
    result = requests.get(url, headers=headers).json()
    return result["items"]
