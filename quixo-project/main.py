from collections import defaultdict
from copy import deepcopy
import inspect
import itertools
import random
import sys
from game import Game, Move, Player
import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move

class QLearningPlayer(Player):
    _Q_table: dict[tuple[int, int], dict[tuple[tuple[int, int], Move], float]] = defaultdict(lambda: defaultdict(float))
    
    def __init__(self) -> None:
        """Create the player for Q learning. Inside it are maintained the training/test state and the trajectory of the moves.
        """
        super().__init__()
        self._is_training: bool = False
        self._trajectory: list[tuple[np.ndarray, tuple[int, int], Move]] = list()
        self._couple_state: bool = True

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        """Take a move randomly if in training mode, using Q table if in evaluation mode.

        Args:
            game (Game): 
                Describe the actual situation of the board.

        Returns:
            tuple[tuple[int, int], Move]: Tuple composed by the coordinates and the `Move` to play on them.
        """
              
        # Symbol to play with. Assinged to the player at each move to be sure that, if the game changed, it is always correct
        self.player = game.get_current_player()  
        
        # In training mode the move is chosen randomly
        if self._is_training:
            # If the state is the same as the last one saved it means that the taken move was not valid and we need to eliminate it from the trajectory
            if len(self._trajectory) > 0 and np.array_equal(self._trajectory[-1][0], game.get_board()):
                self._trajectory.pop()
            # Random choice of the move
            from_pos, move = random.choice(all_possible_coord_moves())
            
            self._trajectory.append((game.get_board(), from_pos, move))
            
        # In playing mode the move is chosen from the Q table
        else:
            state_arr = game.get_board()
            Q_state, rot_num, flip_type = get_board_state(state_arr, self._Q_table, self.player, couple_state=self._couple_state)

            all_good_saved_moves = sorted({k: v for k, v in self._Q_table[Q_state].items() if v > 0}, key= lambda k: self._Q_table[Q_state][k], reverse=True)
            from_pos, move = random.choice(all_possible_coord_moves())
            
            for fp, m in all_good_saved_moves:
                # The move is adjusted to current board and its duability is tested
                fp, m = adjust_coord_move(fp, m, rot_num, flip_type, retrieving=True)
                if simulate_move(self.player, game.get_board(), fp, m) is not None:
                    # If it is good it is taken
                    from_pos, move = fp, m
                    break
            
        return from_pos, move
    
    def _update_Q_table(self, winner: int, learning_rate: float = 0.9, discount_rate: float = 0.75) -> None:
        """Update the Q table considering who won the game.

        Args:
            winner (int): 
                Indicate the symbol of the winner.
            learning_rate (float, optional): 
                Specify the learning rate for the update. Defaults to `0.9`.
            discount_rate (float, optional): 
                Specify the decay rate for the update. Defaults to `0.75`.
        """
        
        for (state_arr, coord, move) in self._trajectory:
            # Retrieve the state from the ones already stored in the table if present
            Q_state, rot_num, flip_type = get_board_state(state_arr, self._Q_table, self.player, couple_state=self._couple_state)
            coord, move = adjust_coord_move(coord, move, rot_num, flip_type, False)
            
            # Update Q table values 
            self._Q_table[Q_state][(coord, move)] += (
                learning_rate * (
                    float(1 if winner == self.player else -1)
                    + discount_rate * self._next_max_state(state_arr)
                    - self._Q_table[Q_state][(coord, move)]
                )
            )
        
        # Reset the trajectory for future training games
        self.__init__()
    
    def _end_training(self) -> None:
        """Switch the agent to evaluation mode: during `make_move` the move is now taken following the Q table.
        """
        self._is_training = False
    
    def _start_training(self) -> None:
        """Switch the agent to training mode: during `make_move` the move is now taken randomly.
        """
        self._is_training = True
    
    def _next_max_state(self, state_arr: np.ndarray) -> float:
        """Return the maximum value reachable in the next current player's move.

        Args:
            state_arr (np.ndarrya): 
                Board state in array form.

        Returns:
            float: Maximum score reachable by the player in its next move.
        """
        values = [.0]
        
        # Search the next state given each possible move done by the opponent
        for coord, move in all_possible_coord_moves():
            next_state = simulate_move(self.player, state_arr, coord, move)
            
            # Update the max value with the ones possibly reachable with the current move combination.
            if next_state is not None:
                q_state, _, _ = get_board_state(next_state, self._Q_table, self.player, couple_state=self._couple_state)
                
                values.extend(self._Q_table[q_state].values())
        
        return max(values) 

    @classmethod
    def save_Q_table(cls, file_name: str = 'Q_table', log_print: bool = False, **kwargs):
        """Save the current Q table in a binary .pkl file.

        Args:
            file_name (str, optional): 
                Output file name. Defaults to `'Q_table'`.
            log_print (bool, optional): 
                If True print the result of the save. Defaults to `False`.
        """
        with open(f'{file_name}.pkl', 'wb') as file:
            pickle.dump(dict(cls._Q_table), file)
        
        if log_print:
            print(f'Q table saved at {file_name}.pkl')
        
    @classmethod
    def load_Q_table(cls, file_name: str = 'Q_table', **kwargs):
        """Load the Q table from the memory, if the file is not found initialize the Q table with default structure.

        Args:
            file_name (str, optional): 
                Input file name. Defaults to `'Q_table'`.
        """
        try:
            with open(f'{file_name}.pkl', 'rb') as file:
                cls._Q_table = defaultdict(lambda: defaultdict(float), pickle.load(file))
            print(f'Q table loaded from {file_name}.pkl')
        except FileNotFoundError:
            print(f'File {file_name}.pkl not found, default Q table will be used')

    @classmethod
    def train_Q_table(cls, *, n_training_games: int = int(1e3), board_generator: Game = Game, table_save_rate: int = int(1e3), learning_rate: float = 0.9, discount_rate: float = 0.75, save_table: bool = True, load_table: bool = True, show_update: bool = True, **kwargs):
        """Train the Q table just by calling this function. All the player istantiated from the `cls` class after this function is called will have the table trained and saved in the file 'file_name'.pkl.

        Args:
            cls:
                `QLearningPlayer` class that called the training.
            n_training_games (int, optional): 
                Number of training games. Defaults to `int(1e3)`.
            board_generator (Game, optional): 
                Class that will generate the game board to play on. Defaults to `Game`.
            table_save_rate (int, optional): 
                Frequency at which the table will be saved in the output file. Defaults to `int(1e3)`.
            learning_rate (float, optional): 
                Specify the learning rate for the update. Defaults to `0.9`.
            discount_rate (float, optional): 
                Specify the decay rate for the update. Defaults to `0.75`.
            save_table (bool, optional): 
                Indicates if after the training we want to save or not the trained table into the file. Defaults to `True`.
            load_table (bool):
                Indicates if the table needs to be loaded from saved files. Defaults to `True`.
            show_update (bool, optional):
                If True show the progression bar of training using tqdm. Defaults to `True`.
                
            **kwargs:
                Args of :func:`load_Q_table` and :func:`save_Q_table`.
        """        
        if load_table:
            cls.load_Q_table(**kwargs)
        
        player1 = cls()
        player2 = cls()
        
        player1._start_training()
        player2._start_training()
        
        for i in tqdm(range(n_training_games))if show_update else range(n_training_games):
            g = board_generator()
            winner = g.play(player1, player2)
            
            player1._update_Q_table(winner, learning_rate, discount_rate)
            player2._update_Q_table(winner, learning_rate, discount_rate)

            if save_table and i%table_save_rate == 0 and i != 0:
                cls.save_Q_table(**kwargs)
        
        if save_table:
            cls.save_Q_table(log_print=True, **kwargs)
    
    @classmethod
    def play_against(cls, *, train: bool = False, opponent: Player = RandomPlayer(), n_evaluation_games: int = int(1e3), print_result: bool = False, **kwargs) -> float:
        """Let the passed player to play against the Q learning agent. Training can be also achieved through this function, otherwise the table will be loaded from file.

        Args:
            cls (QLearningPlayer):
                `QLearningPlayer` class to play against.
            train (bool, optional): 
                Indicates if the table need to be trained or not. Defaults to `False`.
            opponent (Player, optional): 
                Instance of the opponent player's class. Defaults to `RandomPlayer()`.
            n_evaluation_games (int, optional): 
                Number of evaluation games. Defaults to `int(1e3)`.
            print_result (bool, optional): 
                Wheter or not to print the result in the console. Defaults to `False`.
                
            **kwargs:
                show_update (bool, optional):
                    If True show the progression bar of training and testing using tqdm. Defaults to `True`.
                load_table (bool):
                    Indicates if the table needs to be loaded from saved files. Defaults to `True`.
                Args of :func:`cls.train_Q_table` or :func:`cls.load_Q_table`.

        Returns:
            float: Winning percentage of the Q learning player against the passed one.
        """
        show_update = kwargs.get('show_update', True)
        load_table = kwargs.get('load_table', True)
        
        if train:
            cls.train_Q_table(**kwargs)
        elif load_table:
            cls.load_Q_table(**kwargs)

        player1 = cls()
        player2 = opponent
        winner = 0
        
        # Let the first starting player be random each time, then they will be switching the start
        r = random.choice([0, 1])
        
        for i in tqdm(range(n_evaluation_games)) if show_update else range(n_evaluation_games):
            g = Game()
            
            # Change the starting player between games
            winner += 1-g.play(player1, player2) if (i+r)%2 == 0 else g.play(player2, player1)
            
        winning_rate = winner/n_evaluation_games*100
        
        if print_result:
            print(f"{cls.__name__} winning percentage {winning_rate}")
        
        return winning_rate

    @classmethod
    def plot_lr_dr_performance(cls, *, n_values: int = 5, min_lr: float = 0.0, max_lr: float = 1.0, min_dr: float = 0.0, max_dr: float = 1.0, output_dir: str = './quixo-project/images', **kwargs) -> list[tuple[tuple[float, float], float]]:
        """Test `n_values `different values for the learning rate and the discount rate and plot the result. Returns the pair of the best couple of values found.
        
        Plot png image is saved in `output_dir` folder using '`q_learner`__nv_`n_values`__mlr_`min_lr`__Mlr_`max_lr`__mdr_`min_dr`__Mdr_`max_dr`' as naming format.

        Args:
            cls (QLearningPlayer):
                `QLearningPlayer` used to test the values.
            n_values (int, optional): 
                Number of values for each parameter to test on. Defaults to `5`.
            min_lr (float, optional): 
                Minimum value for the learning rate. Defaults to `0.0`.
            min_dr (float, optional): 
                Minimum value for the discount rate. Defaults to `0.0`.
            max_lr (float, optional): 
                Maximum value for the learning rate. Defaults to `1.0`.
            max_dr (float, optional): 
                Maximum value for the discount rate. Defaults to `1.0`.
            output_dir (str, optional):
                Directory where to save the obtained images. Defaults to `'./images'`.
            **kwargs:
                Other args of :func:`QLearningPlayer.play_against` function.
            
        Returns:
            list[tuple[tuple[float, float], float]]: Values of the best performing learning rate and discount rates with the winning percentage associated.
        """
        show_update = kwargs.pop('show_update', False)
        
        # Generation of the values to test
        lr_set = np.linspace(min_lr, max_lr, n_values)
        dr_set = np.linspace(min_dr, max_dr, n_values)
        pair_set = [i for i in itertools.product(lr_set, dr_set)]
        
        win_percentage_set = list()
        
        for lr, dr in tqdm(pair_set):
            win_perc = cls.play_against(train=True, learning_rate=lr, discount_rate=dr, save_table=False, load_table=False, show_update=show_update, **kwargs)
            win_percentage_set.append(win_perc)
            
            cls._Q_table.clear()
            
        x_axis = [i for i in range(n_values**2)]
        
        plt.plot(x_axis, win_percentage_set)
        plt.xticks(x_axis, pair_set, rotation='vertical')
        
        plt.ylabel('Win percentage')
        plt.xlabel('(Learning rate,  Discount rate)')
        plt.grid(True)
        
        plt.title(f'{cls.__name__}, {n_values} values')
        
        plt.savefig(f'{output_dir}/{cls.__name__}__nv_{n_values}__mlr_{min_lr}__Mlr_{max_lr}__mdr_{min_dr}__Mdr_{max_dr}.png', format='png', dpi=300, bbox_inches='tight')
        
        plt.show()
        
        best_indexes = [i for i, val in enumerate(win_percentage_set) if val == max(win_percentage_set)]
        
        return [(pair_set[i], win_percentage_set[i]) for i in best_indexes]

class SelfishQLearningPlayer(QLearningPlayer):
    _Q_table: dict[int, dict[tuple[tuple[int, int], Move], float]] = defaultdict(lambda: defaultdict(float))
    
    def __init__(self) -> None:
        super().__init__()
        self._couple_state = False
    
class MinMaxPlayer(Player):
    _OUTER_RING_MULTIPLIER: int = 1
    _INNER_RING_MULTIPLIER: int = 2
    _CENTRAL_SPOT_MULTIPLIER: int = 3
    _OPPONENT_MULTIPLIER: int = 1
    _WIN_BONUS: int = 30
    
    def __init__(self, *, max_depth: int = 3) -> None:
        """Create a MinMax player with custon maximum search depth.

        Args:
            max_depth (int, optional): 
                Maximum search depth to reduce the time for a move. Defaults to `3`.
        """
        super().__init__()
        self._alphabeta = False
        self._max_depth = max_depth
        
        # Transposition table to reduce the exploration of already seen states for the current player.
        self._transposition_table: dict[tuple[int, int], dict[int, tuple[int, tuple[int, int], Move]]] = dict()
    
    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        """Choose a move according to the board status.

        Args:
            game (Game): Game currently playing containing the board status.

        Returns:
            tuple[tuple[int, int], Move]: Coordinates and `Move` to apply to them to perform the move.
        """
        # If we are using alpha-beta pruning 
        if self._alphabeta:
            alpha = 1-sys.maxsize
            beta = sys.maxsize
        else:
            alpha = beta = None
        
        # Player symbol
        player = game.current_player_idx
        _, coord, move = self._max_move(game.get_board(), player, alpha=alpha, beta=beta, max_depth=self._max_depth)
        
        return coord, move
    
    @staticmethod
    def _evaluate_position(board: np.ndarray, player: int) -> int:
        """Calculate the current board state score from the point of view of player. The scores of the opponent is multiplied by `_OPPONENT_MULTIPLIER`.

        Args:
            board (np.ndarray): Board representation in matrix form.
            player (int): Player to calculate the score for.

        Returns:
            int: Score of the board.
        """
        board = deepcopy(board)
        
        total_score = MinMaxPlayer._row_diag_score(board, player) + MinMaxPlayer._row_diag_score(np.rot90(board), player)
        
        return total_score
    
    @staticmethod
    def _row_diag_score(board: np.ndarray, player: int) -> int:
        """Calculate the score relative to the rows of the board and one diagonal. Pass the rotatation of the board to obtain column and other diagonal score.

        Args:
            board (np.ndarray): Board representation in matrix form.
            player (int): Player to calculato the score of.

        Returns:
            int: Score of the rows and diagonal of the board.
        """
        state_score = 0
        
        # Cicle through all rows
        for r_ind, row in enumerate(board):
            state_score += MinMaxPlayer._vector_score(row, r_ind, player)
            state_score -= MinMaxPlayer._OPPONENT_MULTIPLIER * MinMaxPlayer._vector_score(row, r_ind, 1-player)
            
        # Diagonal score
        state_score += MinMaxPlayer._vector_score(np.diag(board), -1, player)
        state_score -= MinMaxPlayer._OPPONENT_MULTIPLIER * MinMaxPlayer._vector_score(np.diag(board), -1, 1-player)
            
        return state_score
            
    @staticmethod
    def _vector_score(vec: np.ndarray, v_index: int, player: int) -> int:
        """Calculate the score of a vector for the given player.

        Args:
            vec (np.ndarray): Vector representation of a part of the board.
            v_index (int): Index of what row the vector represents. `-1` indicates the diagonal.
            player (int): Symbol to calculate the score of.

        Returns:
            int: Score for the passed vector.
        """
        # The score is computed as 2^(all-1) where all indicates the length of the allined sequence in the row. To this value `WIN_BONUS` is summed if 5 allined symbols are reached. 
        # The score is then multiplied by a factor relative to which positions the tokens belongs to: `OUTER_RING_MULTIPLIER`, external ring; `INNER_RING_MULTIPLIER`, internal ring; `CENTRAL_SPOT_MULTIPLIER`, central spot.
        # To this value is then summed a +`OUTER_RING_MULTIPLIER` or +`INNER_RING_MULTIPLIER` (following the same ring rule) for single tokens in the same vector (case of sequence length of 2 or 3).
        
        # For each sequence length starting from 5 all the way down to 1
        for seq_length in range(5, 0, -1):
            # Generate the sequence to look for
            seq = [player]*seq_length
            # Count the number of times the sequence is present in the vector
            n_seq_found = np.count_nonzero(np.all(np.lib.stride_tricks.sliding_window_view(vec, len(seq)) == seq, axis=1))
            vec_score = 0
            
            if n_seq_found > 0:
                # Calculate the sequence score
                seq_score = 2**(seq_length-1) + (0 if seq_length != 5 else MinMaxPlayer._WIN_BONUS)
                mul = MinMaxPlayer._OUTER_RING_MULTIPLIER
                
                # Token ring position multiplier handling
                if 1 <= v_index <= 3 or v_index == -1:
                    # If occupies the internal ring, multiply per `_INNER_RING_MULTIPLIER`
                    if player in vec[1:4]:
                        mul = MinMaxPlayer._INNER_RING_MULTIPLIER
                    
                    # If occupies the central spot, multiply per `_CENTRAL_SPOT_MULTIPLIER`
                    if v_index == 2 or v_index == -1:
                        if vec[2] == player:
                            mul = MinMaxPlayer._CENTRAL_SPOT_MULTIPLIER
                
                # Apply the multiplier
                vec_score = seq_score * (mul * n_seq_found)
                
                # Add score of single tokens left in the vector
                if n_seq_found == 1 and np.sum(np.where(vec == player, 1, 0)) > seq_length:
                    if 1 <= v_index <= 3 and vec[1] == vec[3] == player:
                        vec_score += MinMaxPlayer._INNER_RING_MULTIPLIER
                    else:
                        vec_score += MinMaxPlayer._OUTER_RING_MULTIPLIER

                # Stop searching for smaller sequences
                break
            
        return vec_score
    
    def _max_move(self, board_arr: np.ndarray, player: int, *, alpha: int|None = None, beta: int|None = None, max_depth: int|None = None) -> tuple[int, tuple[int, int], Move]:
        """Search for the maximum reachable state for the `player` player. If `alpha` and `beta` are defined, alphabeta pruning is used. If `max_depth` is reached the search is truncated.

        Args:
            board_arr (np.ndarray): Array describing the current state to consider.
            player (int): Player's symbol it is using on the board.
            max_depth (int): Maximum depth before stopping the research.
            alpha (int | None, optional): Alpha value for alphabeta pruning. Defaults to `None`.
            beta (int | None, optional): Beta value for alphabeta pruning. Defaults to `None`.

        Returns:
            tuple[int, tuple[int, int], Move]: Reached maximum board value and chosen coordinates and move.
        """            
        # If we reach the end of the tree search or the maximum depth return the player's score in the board.
        if (max_depth is not None and max_depth <= 0) or check_winner(board_arr) != -1:
            return (MinMaxPlayer._evaluate_position(board_arr, player), None, None)

        # Check if the state is already represented in the transposition table. The depth is used as a secondary key in order to not consider as explored a state not fully expanded.
        state, rot_n, flip_type = get_board_state(board_arr, self._transposition_table, player)
        if state not in self._transposition_table or max_depth not in self._transposition_table[state]:
            # Search for the maximum value
            max_value = 1-sys.maxsize
            for coord, move in all_possible_coord_moves(shuffle=True):
                next_board = simulate_move(player, board_arr, coord, move)

                # If the move is applicable check if it improves the score
                if next_board is not None:
                    next_value = self._min_move(next_board, 1-player, alpha=alpha, beta=beta, max_depth=max_depth-1)[0]
                    
                    if next_value > max_value:
                        max_value = next_value
                        max_coord, max_move = coord, move

                        if alpha is not None:
                            alpha = max(alpha, max_value)
                    
                    # Prune if needed
                    if beta is not None and max_value >= beta:
                        break
            
            # Update the transposition table with the result found
            if state not in self._transposition_table:
                self._transposition_table[state] = dict()
            
            # Adjust the move before saving it (according to the manipulations used to find the state in the table)
            max_coord, max_move = adjust_coord_move(max_coord, max_move, rot_n, flip_type, False)
            self._transposition_table[state][max_depth] = (max_value, max_coord, max_move)
            
        # If the state is already present in the table for the current depth
        else:
            # Retrieve and adjust the move
            max_value, max_coord, max_move = self._transposition_table[state][max_depth]
            max_coord, max_move = adjust_coord_move(max_coord, max_move, rot_n, flip_type, True)
            
        return max_value, max_coord, max_move
    
    def _min_move(self, board_arr: np.ndarray, player: int, *, alpha: int|None = None, beta: int|None = None, max_depth: int|None = None) -> tuple[int, tuple[int, int], Move]:
        """Search for the minimum reachable state for the `player` player. If `alpha` and `beta` are defined, alphabeta pruning is used. If `max_depth` is reached the search is truncated.

        Args:
            board_arr (np.ndarray): Array describing the current state to consider.
            player (int): Player's symbol it is using on the board.
            max_depth (int): Maximum depth before stopping the research.
            alpha (int | None, optional): Alpha value for alphabeta pruning. Defaults to `None`.
            beta (int | None, optional): Beta value for alphabeta pruning. Defaults to `None`.

        Returns:
            tuple[int, tuple[int, int], Move]: Reached minimum board value and chosen coordinates and move.
        """
        # If we reach the end of the tree search or the maximum depth return the player's score in the board.
        # In this case the score is not computed for the passed player but for its opponent (the original player for which the algorithm is run)
        if (max_depth is not None and max_depth <= 0) or check_winner(board_arr) != -1:
            return (MinMaxPlayer._evaluate_position(board_arr, 1-player), None, None)
        
        # Check if the state is already represented in the transposition table. The depth is used as a secondary key in order to not consider as explored a state not fully expanded.
        state, rot_n, flip_type = get_board_state(board_arr, self._transposition_table, player)
        if state not in self._transposition_table or max_depth not in self._transposition_table[state]:
            # Search for the minimum value
            min_value = sys.maxsize
            for coord, move in all_possible_coord_moves(shuffle=True):
                next_board = simulate_move(player, board_arr, coord, move)
                
                # If the move is applicable check if it improves the score
                if next_board is not None:
                    next_value = self._max_move(next_board, 1-player, alpha=alpha, beta=beta, max_depth=max_depth-1)[0]
                    
                    if next_value < min_value:
                        min_value = next_value
                        min_coord, min_move = coord, move
                        
                        if beta is not None:
                            beta = min(beta, min_value)
                        
                    # Prune if needed
                    if alpha is not None and min_value <= alpha:
                        break
            
            # Update the transposition table with the result found
            if state not in self._transposition_table:
                self._transposition_table[state] = dict()
            
            # Adjust the move before saving it (according to the manipulations used to find the state in the table)
            min_coord, min_move = adjust_coord_move(min_coord, min_move, rot_n, flip_type, False)
            self._transposition_table[state][max_depth] = (min_value, min_coord, min_move)
                
        # If the state is already present in the table for the current depth
        else:
            # Retrieve and adjust the move
            min_value, min_coord, min_move = self._transposition_table[state][max_depth]
            min_coord, min_move = adjust_coord_move(min_coord, min_move, rot_n, flip_type, True)
            
        return min_value, min_coord, min_move

    @classmethod
    def play_against(cls, *, opponent: Player = RandomPlayer(), n_evaluation_games: int = int(1e3), print_result: bool = False, **kwargs):
        """Let the passed player to play against the MinMax agent. Alpha-beta pruning is enabled depending on the type of :class:`cls`.

        Args:
            cls (MinMaxPlayer):
                `MinMaxPlayer` to play against.
            opponent (Player, optional): 
                Instance of the opponent player's class. Defaults to `RandomPlayer()`.
            n_evaluation_games (int, optional): 
                Number of evaluation games. Defaults to `int(1e3)`.
            print_result (bool, optional): 
                Wheter or not to print the result in the console. Defaults to `False`.
            **kwargs:
                show_update (bool, optional):
                    If True show the progression bar of training and testing using tqdm. Defaults to `True`.
                Other args of `cls` creation.
                
        Returns:
            float: Winning percentage of the MinMax player against the passed one.
        """
        show_update = kwargs.get('show_update', True)
        # Remove unwanted args
        kwargs = {k: v for k, v in kwargs.items() if k == inspect.signature(cls).parameters}
    
        winner = 0
        # Let the first starting player be random each time, then they will be switching the start
        r = random.choice([0, 1])
        
        for i in tqdm(range(n_evaluation_games)) if show_update else range(n_evaluation_games):
            g = Game()
            
            player1 = cls(**kwargs)
            player2 = opponent
            
            # Change the starting player between games
            winner += 1-g.play(player1, player2) if (i+r)%2 == 0 else g.play(player2, player1)
        
        winning_rate = winner/n_evaluation_games*100
        
        if print_result:
            print(f'MinMax winning percentage {winning_rate}')
            
        return winning_rate
        
class AlphaBetaMinMaxPlayer(MinMaxPlayer):
    def __init__(self, *, max_depth: int = 3) -> None:
        super().__init__(max_depth=max_depth)
        self._alphabeta = True

def simulate_move(symbol: int, state_arr: np.ndarray, coord: tuple[int, int], move: Move) -> np.ndarray:
    """Simulate the passed move without modifing the original board state passed.
    
    [Based on :class:`Game` functions]


    Args:
        symbol (int):
            Symbol of the player the move belongs to.
        state_arr (np.ndarray): 
            Board state.
        coord (tuple[int, int]): 
            Coordinates on which to take the move.
        move (Move): 
            `Move` to play on the coordinates.

    Returns:
        np.ndarray: New state obtained with the move.
    """
    g = Game()
    g._board = deepcopy(state_arr)
    is_ok = g._Game__move(coord, move, symbol)
    
    if is_ok:
        return g.get_board()
    
    return None

def check_winner(board: np.ndarray) -> int:
    """Check if there is a winner for the current game board.
    
    [Based on :class:`Game` functions]

    Args:
        board (np.ndarray): 
            Board representation of the game status.

    Returns:
        int: Winner: `0` if player1, `1` if player2, `-1` otherwise.
    """
    g = Game()
    g._board = deepcopy(board)
    return g.check_winner()

def all_possible_coord_moves(*, shuffle: bool = False) -> list[tuple[tuple[int, int], Move]]:
    """Generate all the possible coordinates-move couple considering the limitations of playability of the move (ex. `(0,0)` do not have `Move.TOP`).

    Args:
        shuffle (bool, optional): 
            Decide if shuffle the couples after the generation to let the exploration not stick with the same move. Defaults to `False`.

    Returns:
        list[tuple[tuple[int, int], Move]]: Couples of coordinates and `Move`.
    """
    # The first coordinate is considered to be y during generation and filtering.
    possible_coord = [(x, y) for x in range(5) for y in (range(5) if x in [0, 4] else [0, 4])]
    possible_moves = [Move.TOP, Move.LEFT, Move.BOTTOM, Move.RIGHT]
    
    # Generates all the couples and then filter the impossible ones.
    seq = list(itertools.product(possible_coord, possible_moves))
    seq = list(filter(lambda el: 
                      (el[1] != Move.TOP if el[0][1] == 0 else True)
                      and (el[1] != Move.BOTTOM if el[0][1] == 4 else True)
                      and (el[1] != Move.LEFT if el[0][0] == 0 else True)
                      and (el[1] != Move.RIGHT if el[0][0] == 4 else True), 
                      seq))
    
    if shuffle:
        random.shuffle(seq)
    
    return seq

def get_board_state(state_arr: np.ndarray, search_table: dict, player: int, *, couple_state: bool = True) -> tuple[tuple[int, int]|int, int, int]:
    """Search if the passed board state is already represented in the search table and return the manipulations applied to it in order to find the state and the state itself.
    
    The manipulations can be used to adjust the coordinates and the move of a move associated with the state.

    Args:
        state_arr (np.ndarray): 
            Board state.
        search_table (dict): 
            Table containing all the previously seen states.
        player (int): 
            Player's symbol to evaluate the state for.
        couple_state (bool, optional): 
            If `True` the state is formed by the couple of player's and opponent's scores, otherwise only player's score is used. Defaults to `True`.

    Returns:
        tuple[tuple[int, int]|int, int, int]: Board state converted in couple format `(score player, score opponent)` if `couple_state` is `True` (otherwise single value of player's score), number of rotation and flip type needed to obtain the state.
    """
    state_arr = deepcopy(state_arr)
    state_converter = np.power(2, np.arange(25)).reshape((5,5))
    found = False
    
    # Initialize the state in case no correspondence is found.
    # Keep both the values if couple state is requested, else keep only the first score.
    table_state = get_state_from_board(state_arr, state_converter, player) if couple_state else get_state_from_board(state_arr, state_converter, player)[0]
    for rot_num in range(4):
        for flip_type, rotated_state in enumerate([state_arr, np.fliplr(state_arr), np.flipud(state_arr)]):
            state = get_state_from_board(rotated_state, state_converter, player) if couple_state else get_state_from_board(rotated_state, state_converter, player)[0]
            
            if state in search_table:
                found = True
                table_state = state
                break
            
        if found:
            break
        state_arr = np.rot90(state_arr)
    
    # Reset the transofrmation parameters in case of state not found
    if not found:
        rot_num = 0
        flip_type = 0
    
    return table_state, rot_num, flip_type

def get_state_from_board(board: np.ndarray, state_convert: np.ndarray, player: int) -> tuple[int, int]:
    """Retrieve a unique state for the current board setting given the state convert weights matrix.

    Args:
        board (np.ndarray): 
            Board representation: `0` and `1` player, `-1` None.
        sc (np.ndarray): 
            State convert matrix, same size as board. Specify the weights for the conversion.

    Returns:
        tuple[int, int]: Tuple containing the scores for current player and opponent.
    """
    
    me = np.sum(np.where(board == player, 1, 0) * state_convert)
    adv = np.sum(np.where(board == 1-player, 1, 0) * state_convert)

    return (me, adv) 

def adjust_coord_move(coord: tuple[int, int], move: Move, rot_index: int, flip_type: int, retrieving: bool = False) -> tuple[tuple[int, int], Move]:
    """Adjust the coordinates and the `Move` according to which rotation and flip is found for the current state in the Q table.

    Args:
        coord (tuple[int, int]): 
            Coordinates to adapt.
        move (Move): 
            `Move` to adapt.
        rot_index (int): 
            Number of 90 degrees rotations to apply.
        flip_type (int): 
            Type of flip: `0`, None; `1`, left-right; `2`, up-down.
        retrieving (bool): 
            Indicates if we are converting a move retrieved from the table or not. If `True` the first operations applied are the flips and then the rotations, otherwise the opposite order is used.

    Returns:
        tuple[tuple[int, int], Move]: Tuple containing the adjusted coordinates and `Move`.
    """
    
    # Starting Move
    move_step = [Move.TOP, Move.LEFT, Move.BOTTOM, Move.RIGHT]
    new_move_index = move_step.index(move)
    
    # Starting coordinates
    x = coord[1]
    y = coord[0]
        
    # In case we are adjusting the move for the training we need to first apply the rotation and then the flip (as it is done in the search). 
    # When we are adjusting the move found with the state we are working on we do the opposite.
    for _ in range(2):
        # Apply the flip
        if retrieving:
            if flip_type == 1:
                y = 4-y
            elif flip_type == 2:
                x = 4-x
                
            if (flip_type == 1 and new_move_index % 2 == 1) or (flip_type == 2 and new_move_index % 2 == 0):
                new_move_index = (new_move_index + 2) % 4
                
            rot_index = 4 - rot_index
        
        # Apply the rotations
        else:
            for _ in range(rot_index):
                x, y = 4-y, x
            
            new_move_index = (new_move_index + rot_index) % 4
            
        retrieving = not retrieving
    
    return (y, x), move_step[new_move_index]

def plot_perf(base_player: Player, *, n_evaluation_epochs: int = 10, n_evaluation_games: int = 100, save_plot: bool = True, save_folder: str = './quixo-project/images', **kwargs):
    """Output the plot of `n_evaluation_epochs` epochs of `n_evaluation_games` games. If `save_plot` is `True` save it in `save_folder` with format '`n_evaluation_epochs`epochs_`n_evaluation_games`games_`short_class_name`'.png (`short_class_name` is composed by only the upper case letters of the class name of `base_player` excluding the last one (usually `P`)).

    Args:
        base_player (Player): 
            Player to plot against `RandomPlayer()`.
        n_evaluation_epochs (int, optional): 
            Number of epochs to test. Defaults to `10`.
        n_evaluation_games (int, optional): 
            Number of games for epoch. Defaults to `100`.
        save_plot (bool, optional):
            If `True` save the generated plot into `save_folder`. Defaults to `True`.
        save_folder (bool, optional):
            Folder to save the image to. Defaults to `'./quixo-project/images/'`.
        **kwargs:
            Opponent (Player):
                Instance of `Player` class. If specified, change the opponent to play against.
            Other args of `base_player.play_against()`.
    """
    p = []
    load_table = kwargs.pop('load_table', False)
    kwargs['load_table'] = load_table
    
    for _ in tqdm(range(n_evaluation_epochs)):
        p.append(base_player.play_against(n_evaluation_games=n_evaluation_games, **kwargs))
        # Avoid reloading the table for next games
        kwargs['load_table'] = False
        
    print(p)
    
    x = [i+1 for i in range(n_evaluation_epochs)]
    
    plt.plot(x, p)
    
    plt.ylabel('Win percentage')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.title(f'{base_player.__name__}, {n_evaluation_epochs} epochs {n_evaluation_games} games each')
    
    short_class_name = ''.join(carattere for carattere in base_player.__name__ if carattere.isupper())[:-1]
    
    if save_plot:
        plt.savefig(f'{save_folder}/{n_evaluation_epochs}epochs_{n_evaluation_games}games_{short_class_name}.png', format='png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
if __name__ == '__main__':
    # Q learning learning and discount rate evaluation (already set up for training)
    # print(QLearningPlayer.plot_lr_dr_performance(n_training_games = int(1e4), show_update=True))
    # print(SelfishQLearningPlayer.plot_lr_dr_performance(n_training_games = int(1e3), show_update=True))

    # Q learning training (already trained)
    # QLearningPlayer.play_against(train=True, n_training_games=int(1e6), print_result=True, learning_rate=0.75, decay_rate=0.5, table_save_rate=int(5e4), load_table=False)
    # SelfishQLearningPlayer.play_against(train=True, n_training_games=int(1e6), print_result=True, learning_rate=1.0, decay_rate=0.5, table_save_rate=int(5e4), load_table=False, file_name='Selfish_Q_table')
    
    # Q learning plot of games
    plot_perf(QLearningPlayer, load_table=True, n_evaluation_epochs=10, n_evaluation_games=1000, save_plot=True)
    plot_perf(SelfishQLearningPlayer, load_table=True, n_evaluation_epochs=10, n_evaluation_games=1000, save_plot=True, file_name='Selfish_Q_table')
    
    # Minmax plot of games
    plot_perf(MinMaxPlayer, n_evaluation_epochs=10, n_evaluation_games=1000, save_plot=True, max_depth=3)
    plot_perf(AlphaBetaMinMaxPlayer, n_evaluation_epochs=10, n_evaluation_games=1000, save_plot=False, max_depth=3)
    pass