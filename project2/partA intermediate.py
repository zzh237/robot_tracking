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

warehouse = ['1#2',
             '.#.',
                '..@']
todo = ['1','2']

class DeliveryPlanner:

	def __init__(self, warehouse, todo):
		self.warehouse = warehouse
		self.todo = todo

	def plan_delivery(self):
		moves = []
		return moves


# --------------------
# Code from Unit 4 - Value Program

delta = [[-1, 0],  # go up
		 [0, -1],  # go left
		 [1, 0],  # go down
		 [0, 1],  # go right
		 [-1, -1],  # go upleft
		 [1, -1],  # go downleft
		 [1, 1],  # go downright
		 [-1, 1]]  # go upright
cost_step = [2, 2, 2, 2, 3, 3, 3, 3]


def compute_value(warehouse, destination, dropzone):
	value = [[99 for row in range(len(warehouse[0]))] for col in range(len(warehouse))]
	policy = [[[0, 0] for row in range(len(warehouse[0]))] for col in range(len(warehouse))]
	print "policystart", policy
	value[destination[0]][destination[1]] = 0
	policy[destination[0]][destination[1]] = [0, 0]
	print "policy update destination", policy
	change = True
	print "value", value

	while change:
		change = False

		for x in range(len(warehouse)):
			for y in range(len(warehouse[0])):
				print "xy", [x, y]
				if [x, y] != destination:

					for a in range(len(delta)):
						print "value", value[x][y]
						x2 = x + delta[a][0]
						y2 = y + delta[a][1]
						print "first x2, y2", x2, y2
						# x2, y2 is adjacent to x, y, and x2, y2 need to be the goal, either destination or box, so can decide lift or down
						if [x2, y2] == destination and destination <> dropzone:
							v2 = value[x2][y2] + 4
							if v2 < value[x][y]:
								change = True
								value[x][y] = v2
								policy[x][y] = [10, 10]

						elif [x2, y2] == destination and destination == dropzone:

							v2 = value[x2][y2] + 2
							if v2 < value[x][y]:
								change = True
								value[x][y] = v2
								policy[x][y] = [-10,-10]
						# to make sure that x2, y2 is within the grid and the point isn't occupied

						else:
							print "warehosue2222", warehouse
							print "x2y2", x2, y2
							if x2 >= 0 and x2 < len(warehouse) and y2 >= 0 and y2 < len(warehouse[0]) and (warehouse[x2][y2] == '.'):
								# v2 is the adjacent point for [x, y], now [x, y] is [1, 0] for plan1, and [x2, y2] is the destination[0, 0]
								v2 = value[x2][y2] + cost_step[a]
								print "v2", v2
								print "valuexyyy", value[x][y]
								if v2 < value[x][y]:
									change = True
									# update the 99 of [1, 0] to 1
									value[x][y] = v2
									policy[x][y] = delta[a]
									print "value", value[x][y]
	for i in range(len(value)):
		print value[i], policy[i]
	return value, policy


warehouse = ['1#2',
             '.#.',
                '..@']


todo = ['1','2']

def find_box(box, grid):
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == box:

                return [i,j]
    raise AssertionError


# for i in todo:
#     print find_box(i, warehouse)
def tolist(warehouse):
    cd = []
    for x in range(len(warehouse)):
        ab = []
        for y in range(len(warehouse[0])):
            ab.append(warehouse[x][y])
        cd.append(ab)
    return cd


# def find_box():
# 	for i in range(len(warehouse)):
# 		for j in range(len(warehouse[0])):
# 			if warehouse[i][j] == box:
# 				return (i, j)


# --------------------

# warehouse = [[1, 2, 3],
# 			 [0, 0, 0],
# 			 [0, 0, 0]]
# dropzone = [2, 0]
# todo = [2, 1]


# ------------------------------------------
# plan - Returns cost to take all boxes in the todo list to dropzone
#
# ----------------------------------------

#   ['move 2 1',   # cost = 2
#    'move 1 0',   # cost = 3
#    'lift 1',     # cost = 4
#    'move 2 1',   # cost = 3
#    'down 2 2',   # cost = 2
#    'move 1 2',   # cost = 3
#    'lift 2',     # cost = 4
#    'down 2 2']   # cost = 2


def plan(warehouse, todo):
    cost = 0
    history = []
    count = 0
    steps_i = []
    steps_j = []
    dropzone = find_box('@', warehouse)
    warehouse = tolist(warehouse)

    for box in todo:
        #Step1: the robot starts, the dropzone is the start, and the robot position is the goal, the compute_value takes the goal position
		#locate the robot start position, if the first time, it is at the dropzone, if not, it is at the last time postion
        if count == 0:
            si, sj = dropzone
            print "count", count
            print "steps_i", steps_i
            print "steps_j", steps_j
        else:
            si = steps_i[-1]
            sj = steps_j[-1]
        goal = find_box(box, warehouse)
        print "goal", goal
        i, j = find_box(box, warehouse)
        value, policy = compute_value(warehouse, goal, dropzone)
        print "called value", value
        #the value is always the value for the robot starts
        cost += value[si][sj]
        # release the box position in the warehouse to zero
        warehouse[i][j] = '0'
        print "si, sj", si, sj
        print "policysisj", policy[si][sj]


        # dx = policy[si][sj][0]
        # dy = policy[si][sj][1]
        # si = si + dx
        # sj = sj + dy
        # print "si2, sj2", si, sj
        # print "policysisj2", policy[si][sj]
        while policy[si][sj] <> [10, 10]:
            # print "si sj", si, sj
            # print "policy first", policy
            dx = policy[si][sj][0]
            dy = policy[si][sj][1]
            si = si + dx
            sj = sj + dy
            history.append('move ' + str(si) + ' ' + str(sj))

        if policy[si][sj] == [10, 10]:
            history.append('lift ' + box)
        print "firsthistory", history

        # # Step2: the robot returns back, start position: adjacent position(si, sj), goal position the dropzone
        value, policy = compute_value(warehouse, dropzone, dropzone)
        cost += value[si][sj]
        print "sisjcu", si, sj
        print "policy[si][sj]cu", policy[si][sj][0]
        while policy[si][sj] <> [-10, -10]:
            # print "si sj", si, sj
            # print "policy first", policy
            dx = policy[si][sj][0]
            dy = policy[si][sj][1]
            si = si + dx
            sj = sj + dy
            history.append('move ' + str(si) + ' ' + str(sj))

        if policy[si][sj] == [-10, -10]:
            history.append('down ' + str(dropzone[0]) + ' ' + str(dropzone[1]))


        print "policy after", policy
        print "ni, nj", si, sj
        print "history", history
        print "called second value", value

        count += 1
        #record the robot last steps after finish a one time work
        steps_i.append(si)
        steps_j.append(sj)
    return cost, history

print plan(warehouse, todo)

################# TESTING ##################

# ------------------------------------------
# solution check - Checks your plan function using
# data from list called test[]. Uncomment the call
# to solution_check to test your code.
#
def solution_check(test, epsilon=0.00001):
	answer_list = []

	import time
	start = time.clock()
	correct_answers = 0
	for i in range(len(test[0])):
		user_cost = plan(test[0][i], test[1][i], test[2][i])
		true_cost = test[3][i]
		if abs(user_cost - true_cost) < epsilon:
			print "\nTest case", i + 1, "passed!"
			answer_list.append(1)
			correct_answers += 1
		# print "#############################################"
		else:
			print "\nTest case ", i + 1, "unsuccessful. Your answer ", user_cost, "was not within ", epsilon, "of ", true_cost
			answer_list.append(0)
	runtime = time.clock() - start
	if runtime > 1:
		print "Your code is too slow, try to optimize it! Running time was: ", runtime
		return False
	if correct_answers == len(answer_list):
		print "\nYou passed all test cases!"
		return True
	else:
		print "\nYou passed", correct_answers, "of", len(answer_list), "test cases. Try to get them all!"
		return False


# Testing environment
# Test Case 1
warehouse1 = [[1, 2, 3],
			  [0, 0, 0],
			  [0, 0, 0]]
dropzone1 = [2, 0]
todo1 = [2, 1]
true_cost1 = 9
# Test Case 2
warehouse2 = [[1, 2, 3, 4],
			  [0, 0, 0, 0],
			  [5, 6, 7, 0],
			  ['x', 0, 0, 8]]
dropzone2 = [3, 0]
todo2 = [2, 5, 1]
true_cost2 = 21

# Test Case 3
warehouse3 = [[1, 2, 3, 4, 5, 6, 7],
			  [0, 0, 0, 0, 0, 0, 0],
			  [8, 9, 10, 11, 0, 0, 0],
			  ['x', 0, 0, 0, 0, 0, 12]]
dropzone3 = [3, 0]
todo3 = [5, 10]
true_cost3 = 18

# Test Case 4
warehouse4 = [[1, 17, 5, 18, 9, 19, 13],
			  [2, 0, 6, 0, 10, 0, 14],
			  [3, 0, 7, 0, 11, 0, 15],
			  [4, 0, 8, 0, 12, 0, 16],
			  [0, 0, 0, 0, 0, 0, 'x']]
dropzone4 = [4, 6]
todo4 = [13, 11, 6, 17]
true_cost4 = 41



warehouse = [[1, 19, 2],
			  [0, 19, 0],
			  [0, 0, 0]]
dropzone = [2, 2]
todo = [1, 2]
# testing_suite = [[warehouse1, warehouse2, warehouse3, warehouse4],
#                 [dropzone1, dropzone2, dropzone3, dropzone4],
#                 [todo1, todo2, todo3, todo4],
#                 [true_cost1, true_cost2, true_cost3, true_cost4]]


# solution_check(testing_suite) #UNCOMMENT THIS LINE TO TEST YOUR CODE
# print plan(warehouse, dropzone, todo)

