def print_board(board):
    """Prints the Tic-Tac-Toe board."""
    for row in board:
        print(" | ".join(row))
        print("-" * 9)

def check_win(board, player):
    """Checks if the current player has won."""
    # Check rows
    for row in board:
        if all(s == player for s in row):
            return True
    # Check columns
    for col in range(3):
        if all(board[row][col] == player for row in range(3)):
            return True
    # Check diagonals
    if all(board[i][i] == player for i in range(3)) or \
       all(board[i][2 - i] == player for i in range(3)):
        return True
    return False

def is_board_full(board):
    """Checks if the board is full."""
    return all(cell != ' ' for row in board for cell in row)

def get_player_move(player):
    """Gets the player's move."""
    while True:
        try:
            move = input(f"Player {player}, enter your move (row,col): ")
            row, col = map(int, move.split(','))
            if 0 <= row <= 2 and 0 <= col <= 2:
                return row, col
            else:
                print("Invalid move. Row and column must be between 0 and 2.")
        except ValueError:
            print("Invalid input. Please enter row,col as two numbers separated by a comma.")

def main():
    """Main function to run the Tic-Tac-Toe game."""
    board = [[' ' for _ in range(3)] for _ in range(3)]
    current_player = 'X'

    while True:
        print_board(board)
        row, col = get_player_move(current_player)

        if board[row][col] == ' ':
            board[row][col] = current_player
            if check_win(board, current_player):
                print_board(board)
                print(f"Player {current_player} wins!")
                break
            elif is_board_full(board):
                print_board(board)
                print("It's a draw!")
                break
            current_player = 'O' if current_player == 'X' else 'X'
        else:
            print("That cell is already taken. Try again.")

if __name__ == "__main__":
    main()
