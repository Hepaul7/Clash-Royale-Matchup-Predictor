# Clash Royale Matchup Predictor
NOTE: THIS PROJECT IS STILL IN PROGRESS \
I am making this project on my own, for fun. Especially since I loved this game when I was younger.
```
I'm currently implementing a new way of comparing Matchup
Please ignore all the code in learning and deck prediction
```
# TODO:
1. Use more stats as features from players (Based on Clash Royale API's available information)
   1. Player Challenge Max Wins
   2. Win Rate (Calculate by getting Wins/BattleCount)
   3. Best Season Result (Old Trophy League) [Rank]
2. Fix the neural network
3. Write my own implementation of NB
4. Visualization:
   1. card, good against, plot for each card etc?
   2. 

## Overview:
### Clash Royale:
Clash Royale is a real-time, 1v1 battle against other players. You and your opponents both build a deck of 8 cards. 
The aim is to destroy opponent towers while defending your own. Match time is 3 minutes with 2 minutes extra time. 
Destroying king tower wins the game (unless your opponent destroys your king tower at the same time, resulting draw). 
Otherwise, the player with more damage / crowns wins the game.

### Problem:
The game is not fair sometimes, it's kind of like rock-paper-scissors, where one deck could hard counter another deck.
I am building this model to predict the likelihood of one deck beating another deck. 

### Assumptions:
There are a few assumptions I am making:
  - Both players have the same knowledge on the game (eg. card interactions)
  - Players do not make "stupid" mistakes
  - There aren't any "broken" cards


## Data Collection:
### Collecting:
Since I haven't found any data publicly online I decided to generate my own data.
I generated my own API Token from Clash Royale: https://developer.clashroyale.com/#/

I couldn't find specific league season history for recent months, hence I am using 2022-08, the latest history of top 10,000
players. I navigated to the player history (battle). 
From there, I collected three information. 
- Player Deck
- Opponent Deck
- Outcome of the match (WIN or DEFEAT for player)

Then, I stored the results into a csv file, there are approximately 80,000 entries.

### Cleaning:
I did not consider any battles of the player that were not 1v1 pathOfLegend type.
pathOfLegend is the new "ranked" mode. This is to ensure consistency, where other modes
player could player for "fun", causing many outliers. 


### Imbalanced Data:
The data is imbalanced, where there are more wins than losses.
Therefore, I modified the loss function to take class-imbalance into account.
The cost_sensitive function has an addition constant c+ and c- where c+, c- > 0 to
control tradeoff. 

I am also considering upsampling /downsampling.

## Learning:
### Models:
#### Neural Network:




