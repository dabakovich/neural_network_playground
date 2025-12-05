import logging

from experiments.reinforcement_learning.agent import TicTacToeAgent
from experiments.reinforcement_learning.constants import (
    IS_END_GAME_ON_WRONG_SPOT,
    BoardStringValue,
    BoardValue,
    BoardValueMap,
    GameResult,
    Reward,
)
from experiments.reinforcement_learning.error import GameOverError, SpotTakenError
from experiments.reinforcement_learning.game import Game
from experiments.reinforcement_learning.game_helpers import user_pause
from experiments.reinforcement_learning.statistics import Statistics


def run_game(agents: list[TicTacToeAgent], statistics: Statistics):
    wrong_spots_count = 0
    game = Game()
    x_turn = True
    logging.debug(f"[run_game] Initial board:\n{game}")

    while True:
        agent = agents[0 if x_turn else 1]
        opponent = agents[1 if x_turn else 0]
        board = game.board if x_turn else game.inverted_board
        move = BoardValue.X if x_turn else BoardValue.O

        try:
            spot_index = agent.select_spot_index(board)
            logging.debug(
                f"[run_game] {agent.name} selected '{BoardStringValue.X if x_turn else BoardStringValue.O}' for {spot_index + 1}"
            )

            user_pause()

            if not game.can_make_move(spot_index):
                wrong_spots_count += 1
                if IS_END_GAME_ON_WRONG_SPOT:
                    # Add last step before exception
                    agent.history.add_step(board, spot_index)

                    raise SpotTakenError(index=spot_index)
                else:
                    agent.reinforce_learn_one_move(board, spot_index)
                    logging.debug(f"[run_game] Board after RL one move:\n{game}")
                    continue

            agent.history.add_step(board, spot_index)
            game.next_move(spot_index, move)

            logging.debug(f"[run_game] Board after move:\n{game}")

            x_turn = not x_turn
        except SpotTakenError as e:
            print(f"Wrong spot {e.index + 1 if e.index is not None else ''}")
            print("Ending game...")

            agent.history.finish_game(Reward.WRONG_SPOT, True)
            opponent.history.finish_game(Reward.TIE)

            # Append wrong spot game result if we're not allowing retries
            statistics.append_game_result(GameResult.WRONG_SPOT)

            return

        except GameOverError as e:
            if e.winner is None:
                print("Game over, it's tie")
                agent.history.finish_game(Reward.TIE)
                opponent.history.finish_game(Reward.TIE)

                statistics.append_game_result(GameResult.TIE)
            else:
                logging.info(f"Game over, winner is {BoardValueMap[e.winner]}")
                logging.debug(f"[run_game] Board after end game:\n{game}")

                is_agent_winner = e.winner == move
                agent.history.finish_game(
                    Reward.WIN if is_agent_winner else Reward.LOSE
                )
                opponent.history.finish_game(
                    Reward.WIN if not is_agent_winner else Reward.LOSE
                )

                statistics.append_game_result(
                    agent.name if is_agent_winner else opponent.name
                )

            # Append wrong spots count per game, if we're allowing retries
            statistics.append_wrong_spots_count(wrong_spots_count)
            return
