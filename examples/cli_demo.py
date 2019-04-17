import argparse

from gamegym.games import Goofspiel, RockPaperScissors, TicTacToe, Gomoku
from gamegym.strategy import UniformStrategy
from gamegym.ui.cli import play_in_terminal

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('game', help="Game to play: Gomoku, TicTacToe, Goofspiel, RockPaperScissors / RPS")
    ap.add_argument('N', type=int, default='4', nargs='?', help="Param to the games (if meaningful)")
    ap.add_argument('-t', '--two-players', action="store_true", help='Two human players')
    ap.add_argument('-u', '--uniform', action="store_true", help='Second player uniformly random (default)')
    ap.add_argument('-s', '--symmetrize', action="store_true", help='Second player observes as if he was first')
    ap.add_argument('-n', '--no-colors', action="store_true", help='Disable colors')
    args = ap.parse_args()

    if args.game.lower() == 'gomoku':
        g = Gomoku(args.N, args.N, args.N)
    elif args.game.lower() == 'tictactoe':
        g = TicTacToe()
    elif args.game.lower() == 'goofspiel':
        g = Goofspiel(args.N, scoring=Goofspiel.Scoring.ABSOLUTE)
    elif args.game.lower() in ('rockpaperscissors', 'rps'):
        g = RockPaperScissors()
    else:
        raise Exception('invalid game')

    ad = g.__class__.TextAdapter(g, colors=not args.no_colors, symmetrize=args.symmetrize)
    strats = [None, UniformStrategy()]
    if args.two_players:
        strats[1] = None
    play_in_terminal(ad, strats)
    
if __name__ == '__main__':
    main()
