import numpy as np
import heapq

BOARD_SIZE = 2  # or "manhattan" or "euclidean" or "linear_conflict"

INITIAL_STATE = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
INITIAL_STATE[0][0] = 0
INITIAL_STATE[0][1] = 1
# INITIAL_STATE[0][2] = 3
INITIAL_STATE[1][0] = 2
INITIAL_STATE[1][1] = 3
# INITIAL_STATE[1][2] = 6
# INITIAL_STATE[2][0] = 8
# INITIAL_STATE[2][1] = 7
# INITIAL_STATE[2][2] = 0


GOAL_STATE = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
GOAL_STATE[0][0] = 1
GOAL_STATE[0][1] = 2
GOAL_STATE[0][2] = 3
GOAL_STATE[1][0] = 4
GOAL_STATE[1][1] = 5
GOAL_STATE[1][2] = 6
GOAL_STATE[2][0] = 7
GOAL_STATE[2][1] = 8
GOAL_STATE[2][2] = 0


def hamming_distance(state1, state2):
    return np.sum(state1 != state2) - 1   # element wise comparison kore boolean array return korbe, then okhane 1 0 hishabe dhore sum korbe

def manhattan_distance(state1, state2):
    distance = 0
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if state1[i][j] != 0:
                goal_position = np.argwhere(state2 == state1[i][j])[0]
                distance += abs(i - goal_position[0]) + abs(j - goal_position[1])
    return distance

def euclidean_distance(state1, state2):
    distance = 0
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if state1[i][j] != 0:
                goal_position = np.argwhere(state2 == state1[i][j])[0]
                distance += ((i - goal_position[0]) ** 2 + (j - goal_position[1]) ** 2) ** 0.5
    return distance

def calculateConflict(state, goalState):
    conflicts = 0

    # Calculate conflicts for rows...etakei amra abar column eo use korbo
    for i in range(state.shape[0]):
        #row conflict hishab kori 
        current_row = state[i]  #each row dhorlam
        # print(f"row no {i} :  {current_row}")
        tiles_in_row = []  #hishaber row er je tile gula tar goal row tei ase already tader (value, current y, target y) store korbo
        #oi row er shob element check korbo
        for j in range(state.shape[1]): 
            tile = current_row[j]
            # print(f" state ( {i}, {j} ) , value {tile}")
            if tile != 0: #blank title ta baade    
                target_row, target_col = np.argwhere(goalState == tile)[0]
                # print(f"tile {tile} is in TARGET ( {target_row}, {target_col} )")
                if target_row == i:  # tile is in its goal row
                    # print(f"tile {tile} is in its goal row")
                    tiles_in_row.append((tile, j, target_col))
        # print(f"tiles found correct in current row {i} : {tiles_in_row}")

        # Check for conflicts in the row

        
        for a in range(len(tiles_in_row)):
            for b in range(a + 1, len(tiles_in_row)):
                tile_a, col_a, target_a = tiles_in_row[a]
                tile_b, col_b, target_b = tiles_in_row[b]

                if col_a < col_b and target_a > target_b:
                    conflicts += 1
    return conflicts
    
def linear_conflict_distance(state, goalState):
    distance = 0

    # Calculate Manhattan distance
    manhattanDistance = manhattan_distance(state, goalState)

    #calculate linear conflicts

    row_conflicts = calculateConflict(state, goalState)
    col_conflicts = calculateConflict(state.T, goalState.T) 

    print(f"Row conflicts: {row_conflicts}, Column conflicts: {col_conflicts}")

    distance = manhattanDistance + 2 * (row_conflicts + col_conflicts)

    return distance
    

class BoardState:
    def __init__(self, stateArray2D=None, size=BOARD_SIZE):
        
        self.size = size
        self.priority_value = 0
        self.parent = None  # To keep track of the parent state in the search tree
        self.move_cost = 0  # To keep track of the number of moves made to reach this state

        if stateArray2D is not None:
            self.stateArray2D = np.array(stateArray2D)
        else:
            self.stateArray2D = np.zeros((size, size), dtype=int)

    def __lt__(self, other):
        return self.priority_value < other.priority_value
    def __eq__(self, other):
        return np.array_equal(self.stateArray2D, other.stateArray2D)
    def __hash__(self):
        return hash(tuple(map(tuple, self.stateArray2D)))


    def set_priority_value(self, value):
        self.priority_value = value
    def set_parent(self, parent):
        self.parent = parent

    def is_goal_state(self):
        return np.array_equal(self.stateArray2D, GOAL_STATE)

    def print_state(self):
        for row in self.stateArray2D:
            print(*row)
        
    def generate_neighbors(self):
        neighbors_BoardStates = []
        blank_position = np.argwhere(self.stateArray2D == 0)[0]
        print(f"Zero is at position: {blank_position}")

        adjuscent_positions = [
            (blank_position[0] - 1, blank_position[1]),  # Up
            (blank_position[0] + 1, blank_position[1]),  # Down
            (blank_position[0], blank_position[1] - 1),  # Left
            (blank_position[0], blank_position[1] + 1)   # Right
        ]

        valid_adjuscent_positions = np.array([
            pos for pos in adjuscent_positions
            if 0 <= pos[0] < self.size and 0 <= pos[1] < self.size
        ])

        print(f"Valid positions: \n {valid_adjuscent_positions}")

        for pos in valid_adjuscent_positions:
            new_state_array = np.copy(self.stateArray2D)
            new_state_array[blank_position[0], blank_position[1]] = new_state_array[pos[0], pos[1]]
            new_state_array[pos[0], pos[1]] = 0
            neighbors_BoardStates.append(BoardState(new_state_array))

        print(f"Number of valid  neighbors: {len(neighbors_BoardStates)} and their states:")
        for neighbor in neighbors_BoardStates:
            neighbor.print_state()
            print()

        return neighbors_BoardStates

class N_Puzzle:
    def __init__(self, initial_state=INITIAL_STATE , huristic_function=hamming_distance, goal_state=GOAL_STATE ):
        self.initial_state = initial_state 
        self.goal_state = goal_state 
        self.current_state = BoardState(self.initial_state)
        self.huristic_function = huristic_function
        self.move_count = 0

    def is_solvable(self):
        # Count inversions in the initial state
        inversions = 0
        flat_state = self.initial_state.flatten()
        for i in range(len(flat_state)):
            for j in range(i + 1, len(flat_state)):
                if flat_state[i] != 0 and flat_state[j] != 0 and flat_state[i] > flat_state[j]:
                    inversions += 1
        
        k= self.initial_state.shape[0] 

        if k % 2 == 1:
            if inversions % 2 == 0:
                return True
            else:   
                return False
        else:
            blank_row = self.initial_state.shape[0] - np.argwhere(self.initial_state == 0)[0][0]  # 0 wala shob position retern kore. shetar 1st ta nibo. then tar theke x ta nibo
            if blank_row % 2 == 0 and inversions % 2 == 1:
                return True
            elif blank_row % 2 == 1 and inversions % 2 == 0:
                return True
            else:
                return False
            
    def solve(self):
        if not self.is_solvable():
            print("The puzzle is not solvable.")
            return None
        # implementing A* algorithm 

        #initializing
        start_State = BoardState(self.initial_state)
        self.current_state = start_State
        self.current_state.set_priority_value(0)
        self.current_state.set_parent(None) 

        expanded_nodes = set() #closed list
        open_list = []
        heapq.heappush(open_list, start_State)

        # loop until we find the goal state 
        while open_list:
            current_state = heapq.heappop(open_list)
            expanded_nodes.add(current_state)

            if current_state.is_goal_state():
                path = []
                temp = current_state
                while temp:
                    path.append(temp)
                    temp = temp.parent
                path.reverse()
                print(f"Minimum number of moves = {len(path) - 1}")

                return path
            
            expanded_nodes.add(current_state)  # Add the current state to the closed list
            # explore the neighbors of the current state
            neighbors = current_state.generate_neighbors()
            for neighbor in neighbors:
                if neighbor not in expanded_nodes:
                    #neighbor setup
                    neighbor.set_parent(current_state)
                    neighbor.move_cost = current_state.move_cost + 1
                    huristic_value = self.huristic_function(neighbor.stateArray2D, self.goal_state)
                    neighbor.set_priority_value(neighbor.move_cost + huristic_value)

                    if neighbor in open_list :
                        # Check if the new path to the neighbor is better
                        existing_index = open_list.index(neighbor)
                        existing_neighbor = open_list[existing_index]
                        if neighbor.priority_value < existing_neighbor.priority_value:
                            # Update the priority value and parent
                            open_list[existing_index].set_priority_value(neighbor.priority_value)
                            open_list[existing_index].set_parent(current_state)

                    else:
                        # Add the neighbor to the open list
                        heapq.heappush(open_list, neighbor)

    def print_path(self, path):
        for state in path:
            state.print_state()
            print()
                        
                    
                    

        



sample_board = BoardState(INITIAL_STATE)
sample_board.print_state()
# sample_board.generate_neighbors()
# print("Hamming distance:", hamming_distance(sample_board.stateArray2D, GOAL_STATE))
# print("Manhattan distance:", manhattan_distance(sample_board.stateArray2D, GOAL_STATE))
# print("Is goal state:", sample_board.is_goal_state())
# print("Euclidean distance:", euclidean_distance(sample_board.stateArray2D, GOAL_STATE))
calculateConflict(sample_board.stateArray2D, GOAL_STATE)
print("Linear conflict distance:", linear_conflict_distance(sample_board.stateArray2D, GOAL_STATE))

n_puzzle = N_Puzzle(INITIAL_STATE, hamming_distance, GOAL_STATE)
path = n_puzzle.solve()
if path:
    print("Path to goal state:")
    n_puzzle.print_path(path)
