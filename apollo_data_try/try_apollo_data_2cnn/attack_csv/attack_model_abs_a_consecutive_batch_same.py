import pandas as pd
import csv
import random
from random import randint
import os

def launch_attack(csv_input):
    num_rows = csv_input.shape[0]
    vec = [6, 12, 17]
    # col =  vec[randint(0,2)]
    col = vec[0]
    num_cols = csv_input.shape[1]
    label = list(csv_input)
    seq = list(range(1, num_rows - 1))

    step = 10
    b = [seq[i:i + step] for i in range(0, num_rows - 1, step)]
    b_choose = random.sample(b, 221)
    b_choose.sort()
    print("b_choose: ",b_choose)
    for row1 in b_choose:

        rnd = random.uniform(0.08, 0.5)
        temp = randint(1, 2)
        if temp == 2:
            operation = 1
        else:
            operation = -1

        for row in row1:
            val = csv_input[label[col]].iloc[row]
            # rnd = randint(2, 6)

            if val == 'NAN':
                return
            else:
                csv_input.set_value(row, label[col], str(val + operation * rnd))
                csv_input.set_value(row, label[num_cols - 1], '1')
    return csv_input


root_dir = "./apollo"

csv_file = "interpolated_0to24561_test_7370.csv"
# print("root_dir: ", root_dir)


folders = os.listdir(root_dir)
all_folders = []

for folder in folders:
    if os.path.isdir(os.path.join(root_dir, folder)):
        all_folders.append(os.path.join(root_dir, folder))

# print(all_folders)

for folder in all_folders:
    csv_input = pd.read_csv(os.path.join(folder, csv_file))
    # print("csv_input: ", csv_input)
    csv_input['Anamoly'] = '0'
    csv_input = launch_attack(csv_input)

    # print(csv_input)
    # print(type(folder))
    #  folder=''.join('/interpolated_attack.csv')
    # print(os.path.join(folder, 'interpolated_attack_ABS_center_new.csv'))
    csv_input.to_csv(os.path.join(folder, 'interpolated_test_7370_consecutive_0p08_0p5_batch_same.csv'), index = False)

