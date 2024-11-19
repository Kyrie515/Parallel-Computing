# <center>Othello Project</center>

#### <center>Chengyuan Li</center>

In project Othello, we are asked to implement the Othello game to achieve that we can input a character ('c' or 'h' refers to computer player or human player) and a number (refers to the number of steps we can calculate ahead to get the best move). 

## Algorithm

In this project we use the Negamax algorithm to achieve the requirements.The Negamax algorithm is a variant of the Minimax algorithm, widely used in two-player games to determine the best move for a player by considering all possible moves, evaluating them, and choosing the one that maximizes the player's advantage while minimizing the opponent's. 

In my code, the `Negamax` function recursively explores the game tree for a given board state, depth, and player color. It operates by iterating through all possible moves (determined by the `NeighborMoves` function) that a player can make from the current board state. For each potential move, it applies the move to a copy of the board (`PlaceOrFlip`), recursively calls itself to evaluate the opponent's response (flipping the color and reducing the depth), and then negates the score returned by the recursive call to evaluate the move from the current player's perspective.

The use of `cilk::reducer_max` within a `cilk_for` loop allows the algorithm to explore and evaluate multiple moves in parallel, significantly speeding up the computation. The reducer ensures thread safety when updating the maximum score found across all parallel computations. This parallelization leverages multi-core processors to handle the combinatorial explosion of possible game states, a common challenge in game tree exploration.

Furthermore, the parallel exploration of moves drastically reduces the time required to evaluate deep game trees, making it feasible to consider moves several steps ahead within reasonable time constraints. This depth of foresight is crucial for competitive play, where anticipating and countering the opponent's strategies can make the difference between winning and losing.

In summary, the Negamax algorithm, augmented with Cilk Plus for parallel computation, provides a powerful tool for decision-making in two-player games. It enables a deep analysis of the game state, considering both immediate gains and future possibilities, to make informed decisions that enhance the player's chances of winning. The following code shows the logic of my Negamax algorithm:

```c++
int Negamax(Board *temp_board, int color, int depth, Move *best_move) {
    if (depth < 0) { //base case
        return 0;
    }
    Board neighbors = NeighborMoves(*temp_board, color);
    int row;
    int col;

    cilk::reducer_max<int> max_score;
    cilk_for (int row = 8; row > 0; row--) {
        ull my_neighbor_moves = neighbors.disks[color];
        //find the unchecked row
        for (int i = 8; i > row; i--) {
            my_neighbor_moves >>= 8;
        }
        ull this_row = my_neighbor_moves & ROW8;
        for (int col = 8; col > 0 && this_row; col--) {
            if (this_row & COL8) {
                Move move = {row, col};
                Board new_board = *temp_board;
                if ((new_board.disks[X_BLACK] | new_board.disks[O_WHITE]) & MOVE_TO_BOARD_BIT(move)) {
                    /* this implies that this bit is already occupied
                     * which means this is not a valid move
                     */
                    this_row >>= 1;
                    continue;
                }
                if (IS_MOVE_OFF_BOARD(move)) {
                    this_row >>= 1;
                    continue;
                }
                int temp_flip_self = FlipDisks(move, &new_board, color, 0, 1);
                if (temp_flip_self == 0) {
                    // which means none of the tiles could be flipped
                    this_row >>= 1;
                    continue;
                }
                PlaceOrFlip(move, &new_board, color);
                int temp_score_opponent = -Negamax(&new_board, OTHERCOLOR(color), depth - 1, best_move);
                int score;
                if (temp_score_opponent > 20) {
                    score = temp_flip_self;
                } else {
                    score = temp_flip_self - temp_score_opponent;
                }
                max_score = cilk::max_of(score, max_score);
            }
            this_row >>= 1;
        }
    }
    return max_score.get_value();
}
```



## Experiments

### 1. Cilkview Report

As described above, the most important time lag lies in the part of searching the board for the best move. As the depth increases, this search time will also increase exponentially. The implementation of parallelism here is on this part. I use the cilk_for instead of the for for that loop:

`cilk_for (int row=8; row >=1; row--)`

At each loop, the`ComputerBestMoveAhead()` function will iteratively call itself to perform the selection of best move. Also, the last step where we are in the outer loop is not selected as the part for parallelism. I do not consider the other part of possible parallelism since this loop can already occupy the 8 cores provided fully. Furthermore, other part for possible parallelism may not get an optimistic result, and may add up to more overheads. At each end of the cilk_for function, the synchronization is implicitly implemented. The experimental results with the Cilkview are as follows:

![speedup_analysis](/Users/kyrie/Desktop/2024-comp-422-534-exploratory-search-Kyrie515/speedup_analysis.png)

The graph presents the speedup achieved by increasing the number of processors for parallel computations at different depths. It shows that as the number of processors grows, the speedup also increases, but not in direct proportion, indicating sublinear speedup. This sublinear trend is common in parallel computing, where factors such as communication overhead, synchronization, and the non-parallelizable portion of the task prevent a perfect scale-up with additional processors.

The depth of computation plays a significant role in the degree of speedup, with deeper computations (depth=7) benefiting more from parallelism, as seen by the steeper slope of the corresponding line. This suggests that the computational load at depth=7 is sufficient to make effective use of the additional processors, leading to better speedup. In contrast, shallower computations (depths 1 through 6) exhibit a plateauing effect beyond a certain number of processors, highlighting the diminishing returns of adding more processing units. This behavior indicates that there's an optimal number of processors beyond which the performance gain is minimal, and the efficiency of parallelization does not improve. Therefore, for lower-depth tasks, the investment in more processors may not yield significant performance improvements, whereas higher-depth tasks are more likely to benefit from the increased parallelization.



### 2. Efficiency

For the efficiency plot, it is shown as below:

![efficiency_analysis](/Users/kyrie/Desktop/2024-comp-422-534-exploratory-search-Kyrie515/efficiency_analysis.png)

The graph depicts the efficiency of a computational process as a function of the number of threads used. 

Initially, there are significant fluctuations in efficiency as the number of threads increases, with peaks suggesting moments where the speedup is greater than the number of threads. This could be due to superlinear speedup, which is unusual and may occur when adding more threads leads to better utilization of caches or when the working set of the problem fits better in the collective cache of the processors.

However, the overall trend shows that efficiency drops substantially as the number of threads grows, eventually leveling off to a low-efficiency value. This decline indicates that the addition of threads beyond a certain point does not contribute to a proportional increase in speedup and might even lead to a decrease in performance. Reasons for this could include the overhead of managing more threads, contention for shared resources, or parts of the computation that cannot be effectively parallelized.

### 3. HPC Overview

Use the following command to generate the database.

```
>>>hpcrun -e REALTIME@1000 -t ./othello

>>>hpcstruct -j 4 hpctoolkit-othello-measurements

>>>hpcprof -o hpctoolkit-othello-database hpctoolkit-othello-measurements
```

After check the data in the database, we see the majority cost of time is in the `ComputerBestMoveAhead()` function, especially in the `cilk_for` line.

![截屏2024-02-15 14.38.06](/Users/kyrie/Library/Application Support/typora-user-images/截屏2024-02-15 14.38.06.png)

![截屏2024-02-15 14.39.22](/Users/kyrie/Library/Application Support/typora-user-images/截屏2024-02-15 14.39.22.png)



### 4. Cilkvew table

I have used the `lookahead.sh` to generate the cilkview.1 ... cilkview.7, some of the data is listed below.

| Depth                         | 1    | 2    | 3    | 4    | 5    | 6    | 7   |
| ----------------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Work                          | 15,617,583 | 81,855,999 | 513,981,059 | 8,305,475,959 | 151,456,258,453 | 1,015,828,465,706 | 7,893,543,762,597 |
| Span                          | 10,306,748 | 12,483,898 | 19,512,928 | 50,429,547 | 294,221,147 | 511,225,959 | 574,394,595 |
| Burdened span                 | 30,843,048 | 86,921,418 | 267,138,036 | 1,047,798,593 | 7,033,812,634 | 12,487,129,427 | 15,448,139,908 |
| Parallelism      | 1.52 | 6.56 | 26.34 | 164.69 | 514.77 | 1987.04 | 13742.37 |
| Burdened parallelism          | 0.51 | 0.94 | 1.92 | 7.93 | 21.53 | 81.35 | 510.97 |
| Number of spawns/syncs        | 496 | 43,992 | 343,544 | 5,014,768 | 88,739,320 | 588,428,152 | 4,848,193,136 |
| Average instructions / strand | 4,224 | 620 | 498 | 552 | 568 | 575 | 542 |
| Strands along span | 1,705 | 6081 | 20,345 | 80,865 | 543,033 | 970,553 | 1,202,377 |
| Average instructions / strand on span | 6045 | 2052 | 959 | 623 | 541 |526|477|
| Total number of atomic instructions | 5462 | 45,343 | 343,544 | 5,016,139 | 88,740,785 |588,429,535|4,848,194,461|
| Frame count | 8914 | 93,421     | 71,710 | 10,656,320 | 188,570,993 |1,250,409,761|10,302,410,352|
| Entries to parallel region | 62 | 62 | 62 | 62 | 62 |62|62|
