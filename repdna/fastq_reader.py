import sys
import os

def process(lines=None):
    ks = ['name', 'sequence', 'optional', 'quality']
    return {k: v for k, v in zip(ks, lines)}

def read(fn):
    n = 4
    output = []
    with open(fn, 'r') as fh:
        lines = []
        num_lines =  0
        for line in fh:
            lines.append(line.rstrip())
            if len(lines) == n:
                record = process(lines)
                output.append(record)
                lines = []
            if num_lines >= 10000:
                return output
            #end
            num_lines+=1
    return output
#end