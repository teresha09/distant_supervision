import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--in_file')
parser.add_argument('--out_file')
args = parser.parse_args()
f = open('output/result_pred.txt')
line = f.readlines()[-1]
labels = line.split(" ")
f1 = open(args.in_file)
lines = f1.readlines()
new_lines = []
j = 0
for i in range(len(lines)):
    if i % 3 == 2:
        new_lines.append("{}\n".format(labels[j]))
        j += 1
    else:
        new_lines.append(lines[i])
f2 = open(args.out_file, 'w+')
f2.writelines(new_lines)
f2.close()
