import shutil
import os

origin_path = "/nfs/students/campi/models/GIN_er_30"
destination_path = "/nfs/students/campi/best_models/GIN_er_30"
model = "GIN"
n_seeds = 5

# gin er 50
best_models = {
    "g31": 27,
    "g32": 73,
    "g41": 155,
    "g42": 199,
    "g43": 275,
    "g44": 371,
    "g45": 445,
    "g46": 465,
    "s3": 554,
}

# gin er 10
best_models = {
    "g31": 611,
    "g32": 675,
    "g41": 723,
    "g42": 819,
    "g43": 849,
    "g44": 915,
    "g45": 977,
    "g46": 1043,
    "s3": 1114,
}

# gin er 30
best_models = {
    "Triangle": 2,
    "2-Path": 2,
    "4-Clique": 1, 
    "Chordal cycle": 2,
    "Tailed triangle": 2,
    "3-Star": 2,
    "4-Cycle": 2,
    "3-Path": 3,
    "3-Star not ind.": 2,
}
for task, id in best_models.items():
    original_files = [f"{model}_{task}_{id}_{seed}.pkl" for seed in range(1, n_seeds+1)] # change in the future
    destination_files = [f"{model}_{task}_{seed}.pkl" for seed in range(n_seeds)]
    for i in range(n_seeds):
        shutil.copyfile(os.path.join(origin_path, original_files[i]), os.path.join(destination_path, destination_files[i]))