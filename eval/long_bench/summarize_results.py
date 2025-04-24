import json
import os

result_dir = "../../local/long_bench/pred"
print_dataset_name = True
dataset_names = None

for exp_name in os.listdir(result_dir):
    result_file = os.path.join(result_dir, exp_name, "result.json")
    if not os.path.exists(result_file):
        continue
    with open(result_file, 'r') as f:
        result = json.load(f)

    if print_dataset_name:
        print(",".join([f"{dn}" for dn in result]))
        print_dataset_name = False
        dataset_names = [dn for dn in result]
    row =f"{exp_name},"
    for dataset in dataset_names:
        scores = f"{result[dataset]},"
        row += scores

    print(row)