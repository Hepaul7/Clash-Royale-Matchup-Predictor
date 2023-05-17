# Clash Royale Matchup Predictor
NOTE: THIS PROJECT IS STILL IN PROGRESS
I am making this project on my own, for fun. Especially since I loved this game when I was younger.


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
Since I haven't found any data publically online I decided to generate my own data.
I generated my own API Token from Clash Royale: https://developer.clashroyale.com/#/

I couldn't find specific league season history for recent months, hence I am using 2022-08, the latest history of top 10,000
players. I navigated to the player history (battle). 
From there, I collected three information. 
- Player Deck
- Opponent Deck
- Outcome of the match (WIN or DEFEAT for player)

Then, I stored the results into a csv file, there are approximatley 80,000 entries.


