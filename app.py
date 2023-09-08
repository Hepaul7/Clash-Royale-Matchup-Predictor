from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)
# This long dictionary will be removed, it's only here for convenience
MAP = {'Knight': 0,
       'Archers': 1,
       'Goblins': 2,
       'Giant': 3,
       'P.E.K.K.A': 4,
       'Minions': 5,
       'Balloon': 6,
       'Witch': 7,
       'Barbarians': 8,
       'Golem': 9,
       'Skeletons': 10,
       'Valkyrie': 11,
       'Skeleton Army': 12,
       'Bomber': 13,
       'Musketeer': 14,
       'Baby Dragon': 15,
       'Prince': 16,
       'Wizard': 17,
       'Mini P.E.K.K.A': 18,
       'Spear Goblins': 19,
       'Giant Skeleton': 20,
       'Hog Rider': 21,
       'Minion Horde': 22,
       'Ice Wizard': 23,
       'Royal Giant': 24,
       'Guards': 25,
       'Princess': 26,
       'Dark Prince': 27,
       'Three Musketeers': 28,
       'Lava Hound': 29,
       'Ice Spirit': 30,
       'Fire Spirit': 31,
       'Miner': 32,
       'Sparky': 33,
       'Bowler': 34,
       'Lumberjack': 35,
       'Battle Ram': 36,
       'Inferno Dragon': 37,
       'Ice Golem': 38,
       'Mega Minion': 39,
       'Dart Goblin': 40,
       'Goblin Gang': 41,
       'Electro Wizard': 42,
       'Elite Barbarians': 43,
       'Hunter': 44,
       'Executioner': 45,
       'Bandit': 46,
       'Royal Recruits': 47,
       'Night Witch': 48,
       'Bats': 49,
       'Royal Ghost': 50,
       'Ram Rider': 51,
       'Zappies': 52,
       'Rascals': 53,
       'Cannon Cart': 54,
       'Mega Knight': 55,
       'Skeleton Barrel': 56,
       'Flying Machine': 57,
       'Wall Breakers': 58,
       'Royal Hogs': 59,
       'Goblin Giant': 60,
       'Fisherman': 61,
       'Magic Archer': 62,
       'Electro Dragon': 63,
       'Firecracker': 64,
       'Mighty Miner': 65,
       'Elixir Golem': 66,
       'Battle Healer': 67,
       'Skeleton King': 68,
       'Archer Queen': 69,
       'Golden Knight': 70,
       'Monk': 71,
       'Skeleton Dragons': 72,
       'Mother Witch': 73,
       'Electro Spirit': 74,
       'Electro Giant': 75,
       'Phoenix': 76,
       'Cannon': 77,
       'Goblin Hut': 78,
       'Mortar': 79,
       'Inferno Tower': 80,
       'Bomb Tower': 81,
       'Barbarian Hut': 82,
       'Tesla': 83,
       'Elixir Collector': 84,
       'X-Bow': 85,
       'Tombstone': 86,
       'Furnace': 87,
       'Goblin Cage': 88,
       'Goblin Drill': 89,
       'Fireball': 90,
       'Arrows': 91,
       'Rage': 92,
       'Rocket': 93,
       'Goblin Barrel': 94,
       'Freeze': 95,
       'Mirror': 96,
       'Lightning': 97,
       'Zap': 98,
       'Poison': 99,
       'Graveyard': 100,
       'The Log': 101,
       'Tornado': 102,
       'Clone': 103,
       'Earthquake': 104,
       'Barbarian Barrel': 105,
       'Heal Spirit': 106,
       'Giant Snowball': 107,
       'Royal Delivery': 108}


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/matrix.html')
def matrix():
    interaction_matrix = np.load('interaction_matrix.npy')
    cards = [x for x in MAP.keys()]
    return render_template("matrix.html", cards=cards,
                           interactions=interaction_matrix.tolist())


@app.route('/card_lookup.html', methods=['GET', 'POST'])
def lookup():
    if request.method == 'POST':
        fcard = request.form['fcard']
        scard = request.form['scard']

        interaction_matrix = np.load('interaction_matrix.npy')
        fcard_index = MAP[fcard]
        scard_index = MAP[scard]
        result = interaction_matrix[fcard_index][scard_index]

        cards = [x for x in MAP.keys()]
        return render_template(
            "card_lookup.html",
            cards=cards,
            mapping=MAP,
            fcard=fcard,
            scard=scard,
            result=result
        )

    cards = [x for x in MAP.keys()]
    return render_template(
        "card_lookup.html",
        cards=cards,
        mapping=MAP,
        fcard=None,
        scard=None,
        result=None
    )


@app.route('/deck_comparison.html', methods=['GET', 'POST'])
def deck_comparison():
    return render_template('deck_comparison.html')
    # if request.method == 'POST':
    #     print(0)
    #     a1 = request.form['a1']
    #     a2 = request.form['a2']
    #     a3 = request.form['a3']
    #     a4 = request.form['a4']
    #     a5 = request.form['a5']
    #     a6 = request.form['a6']
    #     a7 = request.form['a7']
    #     a8 = request.form['a8']
    #
    #     b1 = request.form['b1']
    #     b2 = request.form['b2']
    #     b3 = request.form['b3']
    #     b4 = request.form['b4']
    #     b5 = request.form['b5']
    #     b6 = request.form['b6']
    #     b7 = request.form['b7']
    #     b8 = request.form['b8']
    #     interaction_matrix = np.load('interaction_matrix.npy')

        # cards = [x for x in MAP.keys()]
        # return render_template(
        #     "deck_comparison.html",
        #     cards=cards,
        #     mapping=MAP,
        #     a1=a1, a2=a2, a3=a3, a4=a4, a5=a5, a6=a6, a7=a7, a8=a8,
        #     b1=b1, b2=b2, b3=b3, b4=b4, b5=b5, b6=b6, b7=b7, b8=b8
        # )


if __name__ == '__main__':
    app.run()
