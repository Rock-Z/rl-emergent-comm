import re
import os
import argparse
import json

# Regex patterns for the specific JSON structures and '---End--'
pattern1 = re.compile(r'^\{"epoch":\s*\d+,\s*"positional_disent":.*\}$')
pattern2 = re.compile(r'^\{"generalization hold out":.*"epoch":\s*\d+\}$')
pattern_end = re.compile(r'^---End--$')

def main():
    parser = argparse.ArgumentParser(description="Filter log lines: keep epoch 3000 and ---End-- markers.")
    parser.add_argument("input", help="Path to the input log file")
    parser.add_argument("output", help="Path to the output file for filtered results")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.input, 'r') as infile, open(args.output, 'w') as outfile:
        for line in infile:
            line = line.strip()
            if pattern_end.match(line):
                outfile.write(line + '\n')
            elif pattern1.match(line) or pattern2.match(line):
                outfile.write(line + '\n')

if __name__ == "__main__":
    main()
