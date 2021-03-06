#
# === Introduction ===
#
# In this problem, you will build a planner that helps a robot
#   find the best path through a warehouse filled with boxes
#   that it has to pick up and deliver to a dropzone.
# 
# Your file must be called `partA.py` and must have a class
#   called `DeliveryPlanner`.
# This class must have an `__init__` function that takes three 
#   arguments: `self`, `warehouse`, and `todo`.
# The class must also have a function called `plan_delivery` that 
#   takes a single argument, `self`.
#
# === Input Specifications ===
# 
# `warehouse` will be a list of m strings, each with n characters,
#   corresponding to the layout of the warehouse. The warehouse is an
#   m x n grid. warehouse[i][j] corresponds to the spot in the ith row
#   and jth column of the warehouse, where the 0th row is the northern
#   end of the warehouse and the 0th column is the western end.
#
# The characters in each string will be one of the following:
#
# '.' (period) : traversable space. The robot may enter from any adjacent space.
# '#' (hash) : a wall. The robot cannot enter this space.
# '@' (dropzone): the starting point for the robot and the space where all boxes must be delivered.
#   The dropzone may be traversed like a '.' space.
# [0-9a-zA-Z] (any alphanumeric character) : a box. At most one of each alphanumeric character 
#   will be present in the warehouse (meaning there will be at most 62 boxes). A box may not
#   be traversed, but if the robot is adjacent to the box, the robot can pick up the box.
#   Once the box has been removed, the space functions as a '.' space.
# 
# For example, 
#   warehouse = ['1#2',
#                '.#.',
#                '..@']
#   is a 3x3 warehouse. The dropzone is at space (2,2), box '1' is located at space (0,0), 
#   box '2' is located at space (0,2), and there are walls at spaces (0,1) and (1,1). The
#   rest of the warehouse is empty space.
#
# The argument `todo` is a list of alphanumeric characters giving the order in which the 
#   boxes must be delivered to the dropzone. For example, if 
#   todo = ['1','2']
#   is given with the above example `warehouse`, then the robot must first deliver box '1'
#   to the dropzone, and then the robot must deliver box '2' to the dropzone.
#
# === Rules for Movement ===
#
# - Two spaces are considered adjacent if they share an edge or a corner.
# - The robot may move horizontally or vertically at a cost of 2 per move.
# - The robot may move diagonally at a cost of 3 per move.
# - The robot may not move outside the warehouse.
# - The warehouse does not "wrap" around.
# - As described earlier, the robot may pick up a box that is in an adjacent square.
# - The cost to pick up a box is 4, regardless of the direction the box is relative to the robot.
# - While holding a box, the robot may not pick up another box.
# - The robot may put a box down on an adjacent empty space ('.') or the dropzone ('@') at a cost
#   of 2 (regardless of the direction in which the robot puts down the box).
# - If a box is placed on the '@' space, it is considered delivered and is removed from the ware-
#   house.
# - The warehouse will be arranged so that it is always possible for the robot to move to the 
#   next box on the todo list without having to rearrange any other boxes.
#
# An illegal move will incur a cost of 100, and the robot will not move (the standard costs for a 
#   move will not be additionally incurred). Illegal moves include:
# - attempting to move to a nonadjacent, nonexistent, or occupied space
# - attempting to pick up a nonadjacent or nonexistent box
# - attempting to pick up a box while holding one already
# - attempting to put down a box on a nonadjacent, nonexistent, or occupied space
# - attempting to put down a box while not holding one
#
# === Output Specifications ===
#
# `plan_delivery` should return a LIST of moves that minimizes the total cost of completing
#   the task successfully.
# Each move should be a string formatted as follows:
#
# 'move {i} {j}', where '{i}' is replaced by the row-coordinate of the space the robot moves
#   to and '{j}' is replaced by the column-coordinate of the space the robot moves to
# 
# 'lift {x}', where '{x}' is replaced by the alphanumeric character of the box being picked up
#
# 'down {i} {j}', where '{i}' is replaced by the row-coordinate of the space the robot puts 
#   the box, and '{j}' is replaced by the column-coordinate of the space the robot puts the box
#
# For example, for the values of `warehouse` and `todo` given previously (reproduced below):
#   warehouse = ['1#2',
#                '.#.',
#                '..@']
#   todo = ['1','2']
# `plan_delivery` might return the following:
#   ['move 2 1',   # cost = 2
#    'move 1 0',   # cost = 3
#    'lift 1',     # cost = 4
#    'move 2 1',   # cost = 3
#    'down 2 2',   # cost = 2
#    'move 1 2',   # cost = 3
#    'lift 2',     # cost = 4
#    'down 2 2']   # cost = 2
#
# === Grading ===
# 
# - Your planner will be graded against a set of test cases, each equally weighted.
# - If your planner returns a list of moves of total cost that is K times the minimum cost of 
#   successfully completing the task, you will receive 1/K of the credit for that test case.
# - Otherwise, you will receive no credit for that test case. This could happen for one of several 
#   reasons including (but not necessarily limited to):
#   - plan_delivery's moves do not deliver the boxes in the correct order.
#   - plan_delivery's output is not a list of strings in the prescribed format.
#   - plan_delivery does not return an output within the prescribed time limit.
#   - Your code raises an exception.
#
# === Additional Info ===
# 
# - You may add additional classes and functions as needed provided they are all in the file `partA.py`.
# - Upload partA.py to Project 2 on T-Square in the Assignments section. Do not put it into an 
#   archive with other files.
# - Ask questions on Piazza if the directions or specifications are unclear. This is the first time
#   this project has ever been given, so help working out the bugs will be greatly appreciated!

class DeliveryPlanner:
    def __init__(self, warehouse, todo):
        self.warehouse = warehouse
        self.todo = todo
        
    def plan_delivery(self):
        cost = 0
        moves = []
        count = 0
        steps_i = []
        steps_j = []
        dropzone = locate_box('@', self.warehouse)
        self.warehouse = tolist(self.warehouse)
    
        for box in self.todo:
            #Step1: the robot starts, the dropzone is the start, and the robot position is the goal, the min_cost_path takes the goal position
            #locate the robot start position, if the first time, it is at the dropzone, if not, it is at the last time postion
            if count == 0:
                si, sj = dropzone
                #print "count", count
                #print "steps_i", steps_i
                #print "steps_j", steps_j
            else:
                si = steps_i[-1]
                sj = steps_j[-1]
            goal = locate_box(box, self.warehouse)
            #print "goal", goal
            i, j = locate_box(box, self.warehouse)
            value, policy = min_cost_path(self.warehouse, goal, dropzone)
            #print "called value", value
            #the value is always the value for the robot starts
            cost += value[si][sj]
            # release the box position in the self.warehouse to zero
            self.warehouse[i][j] = '0'
            #print "si, sj", si, sj
            #print "policysisj", policy[si][sj]

            # dx = policy[si][sj][0]
            # dy = policy[si][sj][1]
            # si = si + dx
            # sj = sj + dy
            # #print "si2, sj2", si, sj
            # #print "policysisj2", policy[si][sj]
            while policy[si][sj] <> [10, 10]:
                # #print "si sj", si, sj
                # #print "policy first", policy
                dx = policy[si][sj][0]
                dy = policy[si][sj][1]
                si = si + dx
                sj = sj + dy
                moves.append('move ' + str(si) + ' ' + str(sj))
    
            if policy[si][sj] == [10, 10]:
                moves.append('lift ' + box)
            #print "firsthistory", moves
    
            # # Step2: the robot returns back, start position: adjacent position(si, sj), goal position the dropzone
            value, policy = min_cost_path(self.warehouse, dropzone, dropzone)
            cost += value[si][sj]
            #print "sisjcu", si, sj
            #print "policy[si][sj]cu", policy[si][sj][0]
            while policy[si][sj] <> [-10, -10]:
                # #print "si sj", si, sj
                # #print "policy first", policy
                dx = policy[si][sj][0]
                dy = policy[si][sj][1]
                si = si + dx
                sj = sj + dy
                moves.append('move ' + str(si) + ' ' + str(sj))
    
            if policy[si][sj] == [-10, -10]:
                moves.append('down ' + str(dropzone[0]) + ' ' + str(dropzone[1]))
    
    
            #print "policy after", policy
            #print "ni, nj", si, sj
            #print "moves", moves
            #print "called second value", value
    
            count += 1
            #record the robot last steps after finish a one time work
            steps_i.append(si)
            steps_j.append(sj)
        return moves


# --------------------
# All the possible moving situations associated with the robot and the cost

delta = [[-1, 0],  # go up
		 [0, -1],  # go left
		 [1, 0],  # go down
		 [0, 1],  # go right
		 [-1, -1],  # go upleft
		 [1, -1],  # go downleft
		 [1, 1],  # go downright
		 [-1, 1]]  # go upright
delta_cost = [2, 2, 2, 2, 3, 3, 3, 3]


#----------------------------------------------
#calculate the min cost for each position to the goal_index in the warehouse
#---------------------------------------------------------------------------
def min_cost_path(warehouse, goal_index, dropzone):
    value = [[99 for row in range(len(warehouse[0]))] for col in range(len(warehouse))]
    policy = [[[0, 0] for row in range(len(warehouse[0]))] for col in range(len(warehouse))]
    #print "policystart", policy
    value[goal_index[0]][goal_index[1]] = 0
    policy[goal_index[0]][goal_index[1]] = [0, 0]
    #print "policy update goal_index", policy
    change = True
    #print "value", value

    while change:
        change = False

        for x in range(len(warehouse)):
            for y in range(len(warehouse[0])):
                #print "xy", [x, y]
                if [x, y] != goal_index:

                    for a in range(len(delta)):
                        #print "value", value[x][y]
                        x2 = x + delta[a][0]
                        y2 = y + delta[a][1]
                        #print "first x2, y2", x2, y2
                        # x2, y2 is adjacent to x, y, and x2, y2 need to be the goal, either goal_index or box, so can decide lift or down
                        if [x2, y2] == goal_index and goal_index <> dropzone:
                            v2 = value[x2][y2] + 4
                            if v2 < value[x][y]:
                                change = True
                                value[x][y] = v2
                                policy[x][y] = [10, 10]

                        elif [x2, y2] == goal_index and goal_index == dropzone:

                            v2 = value[x2][y2] + 2
                            if v2 < value[x][y]:
                                change = True
                                value[x][y] = v2
                                policy[x][y] = [-10,-10]
                        # to make sure that x2, y2 is within the grid and the point isn't occupied

                        else:
                            #print "warehosue2222", warehouse
                            #print "x2y2", x2, y2

                            if x2 >= 0 and x2 < len(warehouse) and y2 >= 0 and y2 < len(warehouse[0]) and (warehouse[x2][y2] == '.' or warehouse[x2][y2] == '@'):
                                #print "warehouse22 x2 y2", warehouse[x2][y2]
                                # v2 is the adjacent point for [x, y], now [x, y] is [1, 0] for plan1, and [x2, y2] is the goal_index[0, 0]
                                v2 = value[x2][y2] + delta_cost[a]
                                #print "v2", v2
                                #print "valuexyyy", value[x][y]
                                if v2 < value[x][y]:
                                    change = True
                                    # update the 99 of [1, 0] to 1
                                    value[x][y] = v2
                                    policy[x][y] = delta[a]
                                    #print "value", value[x][y]
    #for i in range(len(value)):
        #print value[i], policy[i]
    return value, policy


#Find the index of the box in the warehouse
def locate_box(box, warehouse):
    for i in range(len(warehouse)):
        for j in range(len(warehouse[0])):
            if warehouse[i][j] == box:
                return [i,j]
    raise AssertionError

#--------------------------------------------
#to change the string in the warehouse to list
#---------------------------------------------
def tolist(warehouse):
    cd = []
    for x in range(len(warehouse)):
        ab = []
        for y in range(len(warehouse[0])):
            ab.append(warehouse[x][y])
        cd.append(ab)
    return cd


# ------------------------------------------
# plan - Returns the moves to take all boxes in the todo list to dropzone
#
# ----------------------------------------

###############testing####################
warehouse = ['1..',
            '...',
            '@.2']
todo = ['1', '2']




# warehouse= ['..1.',
#             '..@.',
#             '....',
#             '2...']
# todo = ['1', '2']

plan = DeliveryPlanner(warehouse, todo)
print plan.plan_delivery()



