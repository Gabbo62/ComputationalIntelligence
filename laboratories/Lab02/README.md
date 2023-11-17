### Fixed rules  
Tried to implement a rule that always leaves a number of nim equals to a power of 2 reduced by 1 (as for a simple subtraction game approximated optimal strategy):
 - Only the selection rule is changed, the function  `analize` is changed in order to use the sum of all the remaning sticks instead of `nim_sum`.
 - Running multiple times the match (using `play` function) with different starting player (50% of the times for each player), it seems that the implemented strategy defeats the provided one in the majority of the matches (at least `60%` of the times).
 - Incrementing the number of rows of the game doesn't seem to change the equilibrum.
 - Incrementing the number of games played seems to provide a higher percentage of wins of the implemented rule.

### Evolved stategy
A 1+λ (λ = 100 offsprings) strategy is adopted. The fitness function adopted is `nim_sum`.  
The mutation use a `Gaussian distribution` with mean equals to the actual selected number of sticks to take and the variance starts at 1e-4 and it is tweaked in order to "train" its value against the random agent. After the tweak the obtained strategy is tested against the optimal agent.
 - The structure `Nim_move` encodes the selected move and the fitness value of the offspring.
 - After few tests the winning rate is around `40~45%`.

### References
[`[1]`](https://www.researchgate.net/publication/221330080_Evolving_Winning_Strategies_for_Nim-like_Games) - Mihai Oltean, 2004