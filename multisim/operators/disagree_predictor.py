from typing import List
import pickle
import os
import csv

import numpy as np
from config import TH_PRED

def predict(test,
            problem_name,
            predictor_paths: List[str], 
            threshold =  TH_PRED, 
            # do_log = False, 
            # time_log = None,
            ):
    print(test)
    probs = []
    for predictor in predictor_paths:
        classifier = pickle.load(open(predictor, 'rb'))
        probability_classes = classifier.predict_proba(np.asarray([test]))[0]
        probs.append(probability_classes)
    
    # decide whether agreement or not
    is_disagreement = probs[0][1] > threshold and probs[1][1] > threshold

    # if is_disagreement:
    # print("[RoadMutation] Disagreement detected")
    # if do_log:
    #     log_dis_file = os.getcwd() + os.sep + f'{problem_name}_disagree_predict_{time_log}.csv'
    #     file_existed = os.path.exists(log_dis_file)
    #     with open(log_dis_file, mode = 'a+') as f:
    #         write_to = csv.writer(f)
    #         if not file_existed:
    #             write_to.writerow(["test_input"] + [f"certainty_cl{i}" for i in range(len(predictor_paths))] + ["th_disagree"])
    #         write_to.writerow([test] + [a.tolist() for a in probs] + [is_disagreement])
            
    return is_disagreement, probs