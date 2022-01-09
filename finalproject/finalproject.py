import sys
# will be called like this `python finalproject.py input_file`
#  input_file: name of one of the 10 provided test files

# must output file named prediction.txt
#   prediction.txt must contain next 60 predictions (x,y)
#   predictions must be same format as input file
#   1 minute time limit to output all 10 predictions (won't matter b/c our answers will be pre-computed)


infile = sys.argv[1]

with open(infile) as inf:
    with open('prediction.txt', 'w+') as outf:
        for line in inf: outf.writelines(line)
