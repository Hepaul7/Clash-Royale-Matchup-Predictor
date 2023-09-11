"""
This file contains code that use the functions from royale_api.statsroyale
These are all code relating to data processing, cleaning, and reformatting.
"""
from typing import *
import csv
import numpy as np
import royale_api.statsroyale as stats

SEASON = "2023-05"
MATRIX_PATH = "../matrix_updated.npy"
PATH = "../data_updated.csv"
BATTLE_TYPE = "pathOfLegend"
INTERACTION_PATH = "../interaction_matrix.npy"
NUM_CARDS = 109


def map_cards() -> Dict[str, int]:
    """
    Create a dict that stores card names to an index for encoding
    :return: a dict that stores card names to an index for encoding
    """
    cards = stats.get_all_cards()
    return {card["name"]: i for i, card in enumerate(cards)}


def create_interaction_map() -> Dict[str, Dict[str, int]]:
    """
    Create a dict for each of the cards in the format of Dict[str, Dict[str, int]]
    Outer-key: A card name, Outer-value: Another Dict
    Inner-key: A card name, Inner-Value: An int, representing the net-win
    Initialize all net-win values to 0
    """
    cards = stats.get_all_cards()
    return {card["name"]: {in_card["name"]: 0 for j, in_card in enumerate(cards)
                           if in_card["name"] != card["name"]} for i, card in enumerate(cards)}


# load the data from the CSV file
def load_data(batch: Optional[int] = None) -> np.ndarray:
    """ Load the data from the CSV file and convert it to a numpy matrix.

    Vectorize the data,  if there are 109 cards in the game, the input vector would have 219 indices
    (0 to 218), where from 0 to 108 represents vectors of the first player, 109 to 217 represents
    vectors of second player and 218 represents the outcome if the first players deck won

    :return: a ndarray of the data
    """
    vocab = map_cards()
    print(vocab)
    matrix = np.empty((0, 21), int)
    with open('../data_updated.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        # keep track of a matrix of vectors and labels
        curr_batch = 0
        for row in reader:
            print(row)
            if batch is not None:
                if curr_batch == batch:
                    break
            curr_batch += 1
            label = 1 if row[6] == "True" else 0
            blue_cards = eval(row[4])
            red_cards = eval(row[5])

            # vectorize the cards
            blue_vector = np.zeros(8)
            red_vector = np.zeros(8)

            for i in range(len(blue_cards)):
                print(vocab[blue_cards[i]])
                blue_vector[i] = vocab[blue_cards[i]]

            for i in range(len(red_cards)):
                red_vector[i] = vocab[red_cards[i]]

            # combine vectors and label
            deltas = np.zeros(4)
            deltas[0] = int(row[0])
            deltas[1] = int(row[1])
            deltas[2] = int(row[2])
            deltas[3] = int(row[3])
            # deltas[0] = int(row[1])
            # deltas[1] = int(row[2])
            # deltas[2] = int(row[3])
            print(deltas)

            vector = np.concatenate((blue_vector, red_vector))
            print(vector)
            vector = np.append(deltas, vector)
            vector = np.append(vector, label)
            matrix = np.append(matrix, [vector], axis=0)
    print(matrix)
    return matrix


def battle_cleanup(battle: dict) -> (List, List, int):
    """ Given the battle dictionary from Clash Royale's API, return the
    cards the player used, the opponents used, and a
    indicator (1 if player won, 0 if opponent won)

    Note: Its typical to use blue for player, red for opponent in CR
    """
    # get all the cards
    blue_cards = [x["name"] for x in battle["team"][0]["cards"]]
    red_cards = [x["name"] for x in battle["opponent"][0]["cards"]]
    # get the winner (1 if the player won, 0 if opponent wins)
    winner = battle["team"][0]["crowns"] > battle["opponent"][0]["crowns"]
    return blue_cards, red_cards, winner


def fill_interaction_map() -> Dict[str, Dict[str, int]]:
    """
    Fill the interaction map based on recent battles of top 100 players in 2023-05
    note: the code is a bit ugly, ill fix it.
    """
    mapping = create_interaction_map()
    rankings = stats.get_season_ranking(SEASON, 900)["items"]
    for i in range(1, len(rankings)):
        player = rankings[i]
        tag = player["tag"][1:]
        log = stats.get_battle_logs(tag)
        if log is not None:
            for battle in log:
                if battle["type"] == BATTLE_TYPE:
                    blue_cards, red_cards, winner = battle_cleanup(battle)
                    count = 1 if winner == 1 else -1

                    for blue in blue_cards:
                        for red in red_cards:
                            if blue != red:
                                mapping[blue][red] += count
    return mapping


def interaction_to_matrix() -> np.ndarray:
    """ Convert the map into a Matrix
    """
    interaction_matrix = np.zeros((NUM_CARDS, NUM_CARDS))
    interactions = fill_interaction_map()
    card_index = map_cards()
    print(card_index)
    for card, interactions in interactions.items():
        for other_card, net_win in interactions.items():
            interaction_matrix[card_index[card]][card_index[other_card]] = net_win

    np.save(INTERACTION_PATH, interaction_matrix)
    print(interaction_matrix)
    return interaction_matrix


def process_battle_logs() -> None:
    """ Create a CSV file that stores the battle logs of the top 10000 players in the current season

    update: will this player beat the opponent?
    input: delta pb (player - opponent),
    delta gc max wins, delta gt wins,
    delta best PoL rank: (player - opponent), however,
    if player unranked, opponent ranked: -10000
    if player ranked, opponent unranked: 10000
    delta challenge max wins (player - opponent)
    """
    # loop over players and store their battle logs in a CSV file
    with open(PATH, "a", newline="") as csvfile:
        write = csv.writer(csvfile)
        rankings = stats.get_season_ranking(SEASON, 9999)["items"]
        for i in range(500, len(rankings)):  # should iterate 9999 times
            player = rankings[i]
            # remove hashtag from player tag
            tag = player["tag"][1:]
            log = stats.get_battle_logs(tag)
            if log is not None:
                for battle in log:
                    # check if the battle is a 1v1 battle
                    if battle["type"] == BATTLE_TYPE:
                        blue_cards, red_cards, winner = battle_cleanup(battle)

                        opponent_tag = battle['opponent'][0]['tag'][1:]

                        player_pb = stats.get_player_pb(tag)
                        opponent_pb = stats.get_player_pb(opponent_tag)

                        player_max_win = stats.get_max_wins(tag)
                        opponent_max_win = stats.get_max_wins(opponent_tag)

                        player_best_rank = stats.get_pol_best_rank(tag)
                        opponent_best_rank = stats.get_pol_best_rank(opponent_tag)

                        player_gt_wins = stats.get_amount_gt(tag)
                        opponent_gt_wins = stats.get_amount_gt(opponent_tag)

                        if opponent_pb is None or player_pb is None:
                            continue

                        delta_pb = player_pb - opponent_pb
                        delta_max_win = player_max_win - opponent_max_win
                        if player_best_rank == 0 and opponent_best_rank == 0:
                            delta_rank = 0
                        elif player_best_rank == 0:
                            delta_rank = -10000
                        elif opponent_best_rank == 0:
                            delta_rank = 10000
                        else:
                            delta_rank = -1 * (player_best_rank - opponent_best_rank)

                        delta_gt = player_gt_wins - opponent_gt_wins

                        print([delta_pb, delta_max_win, delta_rank,
                               delta_gt, blue_cards, red_cards, winner])
                        # write to CSV
                        write.writerow([delta_pb, delta_max_win, delta_rank,
                                        delta_gt, blue_cards, red_cards, winner])
    csvfile.close()


def save_matrix(matrix):
    np.save(MATRIX_PATH, matrix)
