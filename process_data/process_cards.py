from typing import Dict
import csv
import numpy as np
import royale_api.statsroyale as stats


# create a dict that stores card names to an index for encoding
def map_cards() -> Dict[str, int]:
    """
    Create a dict that stores card names to an index for encoding
    :return: a dict that stores card names to an index for encoding
    """
    cards = stats.get_all_cards()
    return {card["name"]: i for i, card in enumerate(cards)}


# load the data from the CSV file
def load_data(batch: int) -> np.ndarray:
    """ Load the data from the CSV file
    Vectorize the data,  if there are 109 cards in the game, the input vector would have 219 indices
    (0 to 218), where from 0 to 108 represents vectors of the first player, 109 to 217 represents
    vectors of second player and 218 represents the outcome if the first players deck won
    :return: train, validation and test sets
    """
    vocab = map_cards()
    matrix = np.empty((0, 2 * len(vocab) + 1), int)
    with open('../data.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        # keep track of a matrix of vectors and labels
        curr_batch = 0
        for row in reader:
            if curr_batch == batch:
                break
            curr_batch += 1
            label = 1 if row[0] == "True" else 0
            blue_cards = eval(row[1])
            red_cards = eval(row[2])

            # vectorize the cards
            blue_vector = np.zeros(len(vocab))
            red_vector = np.zeros(len(vocab))

            for card in blue_cards:
                blue_vector[vocab[card]] += 1
            for card in red_cards:
                red_vector[vocab[card]] += 1

            # combine vectors and label
            vector = np.concatenate((blue_vector, red_vector))
            vector = np.append(vector, label)
            matrix = np.append(matrix, [vector], axis=0)
            # print(vector)
            # print(matrix.shape)
    return matrix
