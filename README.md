Label, Description,Feature Engineering Idea
- **tourney_id**
  - *Event Prestige*: Group by ID to see if a player performs better at specific recurring tournaments.
- **surface**
  - *Surface Specialist*: Encode this! A player might have a 40% win rate on Hard but 80% on Clay.
- **tourney_level**
  - *Big Stage Factor*: Calculate a "Pressure Rating." Some players crumble in 5-set Slams but dominate 3-set ATP 250s.
- **tourney_date**
  - *Seasonality*: Convert to month. Does the player peak during the "European Clay Swing"?
- **draw_size**: Number of players in the event.
  - *Fatigue Proxy*: Larger draws mean more matches played to reach the final.

- **hand**
  - *Lefty Advantage*: Create a boolean is_lefty_match. Lefties often have a serve advantage that bothers righties.
- **winner_ht / loser_ht**
  - *Serve Potential*: Calculate height_diff. Taller players generally have higher ace counts but might move slower.
- **winner_ioc / loser_ioc**
  - *Home Court*: Does the player win more when ioc matches the tournament’s location?
- **winner_age / loser_age**
  - *Experience Gap*: winner_age - loser_age. Is the "old guard" beating the "next gen"?
- **winner_rank (winner_rank_points) vs. loser_rank (loser_rank_points)**
  - *Upset Metric*: Calculate the log-difference in rank points. This is a massive predictor for win probability.


- **ace / df**
  - *Serve Quality*:​ the control ratio ace/df
- **svpt**: Use this as the denominator for most serve stats.
- *1st Serve %* 1stIN/svpt. Crucial for consistent pressure.
- *Power Factor*: 1stWon​/1stIn. High % here means a heavy, unreturnable serve.
- *Vulnerability*: 2ndWon​/svpt−1stIn. If this is low, the player is easy to break.
- *Hold Rate*: Points won per service game.
- *Clutch Factor*: bpSaved/bpFaced High % shows mental toughness under pressure.






# Tennis ATP Match Dataset - Column Descriptions

| Column | Description |
|---|---|
| `tourney_id` | Unique tournament identifier (`year-tournament_code`). |
| `tourney_name` | Name of the tournament. |
| `surface` | Court surface type (`Hard`, `Clay`, `Grass`, `Carpet`). |
| `draw_size` | Number of players in the tournament draw. |
| `tourney_level` | Tournament category/level (`G`=Grand Slam, `M`=Masters, `A`=ATP, `D`=Davis Cup.). |
| `tourney_date` | Tournament start date in `YYYYMMDD` format. |
| `match_num` | Match number within the tournament. |
| `winner_id` | Unique player ID of the winner. (eg. `104676`)|
| `winner_seed` | Tournament seed of the winner. (eg. `5.0` or `NaN`) |
| `winner_entry` | Entry type of the winner (`Q`=Qualifier, `WC`=Wildcard, `LL`=Lucky Loser, `SE`=Special Exempt). |
| `winner_name` | Full name of the winner. (eg. `Casper Ruud`)|
| `winner_hand` | Dominant hand of the winner (`R`=Right, `L`=Left, `U`=Unknown). |
| `winner_ht` | Height of the winner in centimeters. (eg. `188.0`)|
| `winner_ioc` | Country code (IOC format) of the winner. (eg. `FRA` or `ITA`)|
| `winner_age` | Age of the winner at match time. (eg. `27.[0-9]`) |
| `loser_id` | Unique player ID of the loser. |
| `loser_seed` | Tournament seed of the loser. |
| `loser_entry` | Entry type of the loser. |
| `loser_name` | Full name of the loser. |
| `loser_hand` | Dominant hand of the loser. |
| `loser_ht` | Height of the loser in centimeters. |
| `loser_ioc` | Country code (IOC format) of the loser. |
| `loser_age` | Age of the loser at match time. |
| `score` | Final match score. (eg. `6-4 7-6(2)` or `6-4 6-2 RET`)|
| `best_of` | Number of sets required to win the match (`3` or `5`). |
| `round` | Tournament round (`RR`, `R128`, `R64`, `R32`, `QF`, `SF`, `F`). |
| `minutes` | Match duration in minutes. (eg. `124.0` or `61.0`) |
| `w_ace` | Number of aces served by the winner. |
| `w_df` | Number of double faults committed by the winner. |
| `w_svpt` | Total service points played by the winner. |
| `w_1stIn` | Number of first serves made by the winner. |
| `w_1stWon` | Number of points won on first serve by the winner. |
| `w_2ndWon` | Number of points won on second serve by the winner. |
| `w_SvGms` | Number of service games played by the winner. |
| `w_bpSaved` | Number of break points saved by the winner. |
| `w_bpFaced` | Number of break points faced by the winner. |
| `l_ace` | Number of aces served by the loser. |
| `l_df` | Number of double faults committed by the loser. |
| `l_svpt` | Total service points played by the loser. |
| `l_1stIn` | Number of first serves made by the loser. |
| `l_1stWon` | Number of points won on first serve by the loser. |
| `l_2ndWon` | Number of points won on second serve by the loser. |
| `l_SvGms` | Number of service games played by the loser. |
| `l_bpSaved` | Number of break points saved by the loser. |
| `l_bpFaced` | Number of break points faced by the loser. |
| `winner_rank` | ATP ranking of the winner at match time. |
| `winner_rank_points` | ATP ranking points of the winner. |
| `loser_rank` | ATP ranking of the loser at match time. |
| `loser_rank_points` | ATP ranking points of the loser. |
| `year` | Year of the tournament/match. |