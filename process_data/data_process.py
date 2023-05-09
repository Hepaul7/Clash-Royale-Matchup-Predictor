import csv
import royale_api.statsroyale as stats
SEASON = "2022-08"


# loop over players and store their battle logs in a CSV file
with open("../data.csv", "a", newline="") as csvfile:
    write = csv.writer(csvfile)

    rankings = stats.get_season_ranking(SEASON)["items"]
    print(rankings)
    for i in range(570, len(rankings)):  # loop starting from rank 571
        player = rankings[i]
        print(player)
        # remove hastag from player tag
        tag = player["tag"][1:]
        log = stats.get_battle_logs(tag)
        print(log)
        if log is not None:
            print("DONE")
            for battle in log:
                # check if the battle is a 1v1 battle
                if battle["type"] == "pathOfLegend":
                    # get all the cards
                    blue_cards = [x["name"] for x in battle["team"][0]["cards"]]
                    red_cards = [x["name"] for x in battle["opponent"][0]["cards"]]
                    # get the winner
                    winner = battle["team"][0]["crowns"] > battle["opponent"][0]["crowns"]
                    # write to CSV
                    write.writerow([winner, blue_cards, red_cards])


