import csv
import royale_api.statsroyale as stats

SEASON = "2022-08"
PATH = "../data.csv"
BATTLE_TYPE = "pathOfLegend"


def battle_cleanup(battle: dict):
    """ Clean up
    """
    # get all the cards
    blue_cards = [x["name"] for x in battle["team"][0]["cards"]]
    red_cards = [x["name"] for x in battle["opponent"][0]["cards"]]
    # get the winner
    winner = battle["team"][0]["crowns"] > battle["opponent"][0]["crowns"]
    return blue_cards, red_cards, winner


def process_data() -> None:
    """ Create a CSV file that stores the data of the top 10000 players in the current season
    """
    # loop over players and store their battle logs in a CSV file
    with open(PATH, "a", newline="") as csvfile:
        write = csv.writer(csvfile)
        rankings = stats.get_season_ranking(SEASON)["items"]
        for i in range(1, len(rankings)):  # should iterate 9999 times
            player = rankings[i]
            # remove hashtag from player tag
            tag = player["tag"][1:]
            log = stats.get_battle_logs(tag)
            if log is not None:
                for battle in log:
                    # check if the battle is a 1v1 battle
                    if battle["type"] == BATTLE_TYPE:
                        blue_cards, red_cards, winner = battle_cleanup(battle)
                        # write to CSV
                        write.writerow([winner, blue_cards, red_cards])
