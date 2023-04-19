import json
import os
import pandas as pd


def get_attack_json():
    """
    get the current path
    """
    current_path = os.path.abspath(os.path.dirname(__file__))
    """
    get the data path
    """
    data1 = os.path.abspath(os.path.join(current_path, '../data/data1attack.json'))
    data2 = os.path.abspath(os.path.join(current_path, '../data/data2attack.json'))
    with open(data1, 'r') as f:
        china = json.load(f)

    with open(data2, 'r') as f:
        paris = json.load(f)

    # print(china)
    # print(paris)

    return china, paris


if __name__ == '__main__':
    get_attack_json()