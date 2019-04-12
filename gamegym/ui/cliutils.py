import termcolor

def draw_board(board, symbols, colors=None):
    if colors:
        symbols = [termcolor.colored(s, c) for s, c in zip(symbols, colors)]

    lines = ["  " + "".join(str((i + 1) % 10) for i in range(len(board[0])))]
    lines += ["{} ".format(i % 10 + 1) + "".join(symbols[x] for x in l)
              for i, l in enumerate(board)]
    return "\n".join(lines)