import sys
import time
import subprocess
import os
import numpy as np
import os.path as osp
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('script_file')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    scripts = open(args.script_file).read().split("\n")
    rank_list = []
    error_flag=False
    for script in scripts:
        print("checking ", script)
        variables = script.split(" ")
        conf_flag = False
        for idx, var in enumerate(variables):
            if "configs" in var:
                if not os.path.exists(var):
                    error_flag = True
                    print("can not find config:", var)
                break
    if error_flag is False:
        print("valid scripts")



