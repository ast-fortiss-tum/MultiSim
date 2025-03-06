import json
from math import inf
from typing import List, Tuple
import random
import os
from features.plot_map import plot_map_of_elites
from opensbt.utils.encoder_utils import NumpyEncoder
import numpy as np
from config import CRITICAL_XTE
import logging as log 

class FeatureMap(object):
    def __init__(self, 
                feature_names, 
                bound_up, 
                bound_low, 
                interval,
                min_fitness = 0,
                max_fitness = 1,
                failure_threshold =  - CRITICAL_XTE,
                combine_strategy = "optimal"
        ):
        self.feature_names = feature_names
        self.bound_low = bound_low
        self.bound_up = bound_up
        self.interval = interval
        self.min_fitness = min_fitness
        self.max_fitness = max_fitness 
        self.data = {}
        self.data_all = {}
        self.data_clean = {}
        self.failure_threshold = failure_threshold
        self.input_data_all = {}
        self.combine_strategy = combine_strategy

        assert len(self.feature_names) == len(self.bound_low), f"{len(self.feature_names)} != {len(self.bound_low)}"
        assert len(self.bound_low) == len(self.bound_up)
        assert len(self.interval) == len(self.bound_up)

    def data_to_json(self):
        dmp = {str(k): v for k, v in self.data.items()}
        return json.dumps(dmp,
                allow_nan=True, 
                indent=4,
                cls=NumpyEncoder)

    def data_clean_to_json(self):
        dmp = {str(k): v for k, v in self.data_clean.items()}
        return json.dumps(dmp,
                allow_nan=True, 
                indent=4,
                cls=NumpyEncoder)
                
    def data_all_to_json(self):
        dmp = {str(k): v for k, v in self.data_all.items()}
        return json.dumps(dmp,
                allow_nan=True, 
                indent=4,
                cls=NumpyEncoder)
    
    def input_data_all_to_json(self):
        dmp = {str(k): v for k, v in self.input_data_all.items()}
        return json.dumps(dmp,
                allow_nan=True, 
                indent=4,
                cls=NumpyEncoder)
    
    def update_map(self, feature_values: Tuple[float], fitness: float, input_values: List[float]):
        assert(len(feature_values) == len(self.bound_low))
        data = self.data
        bins = tuple([self.get_bin(feature_value=feature, feature_id=id) for id, feature in enumerate(feature_values)])
        
        # real data
        self.data_clean[str(feature_values)] = fitness
        
        # abstracted with all fitness
        if bins in self.data_all:
            self.data_all[bins].append(fitness)
            self.input_data_all[bins].append(input_values)
        else:
            self.data_all[bins] = [fitness]
            self.input_data_all[bins] = [input_values]
        
        # abstracted data (the one we see in the map)
        # use worst fitness
        if bins in data:
            if self.combine_strategy == "optimal":
                if data[bins] > fitness:
                    #replace
                    data[bins] = fitness
            else:
                # average
                data[bins] = np.mean(self.data_all[bins])
        else:
            data[bins] = fitness

        n_all = 0
        for i,j in self.input_data_all.items():
            n_all += len(j)

    def preserve_int(self, tuple_values):
        t = ()
        for v in tuple_values:
            v_new = int(v) if int(v) == v else v
            t = t + (v_new,)
        return t

    def plot_map(self, filepath = "./features/", filename = None, title_suffix = "",
                 iterations = -1):

        if filename is not None:
            filename = os.path.basename(filename)
        # we need to map back
        data_raw = {}
        for features, fitness in self.data.items():
            feature_interpolated = self.preserve_int(
                tuple([features[id] * self.interval[id] + self.bound_low[id] for \
                        id in range(len(self.feature_names))])
            )
            data_raw[feature_interpolated] = fitness
        
        # fill with empty values
        for k in np.arange(self.bound_low[0], self.bound_up[0], self.interval[0]):
            for l in np.arange(self.bound_low[1], self.bound_up[1], self.interval[1]):
                if (k,l) not in data_raw:
                    # avoid too many digits after comma; write int number of now digits after comma
                    k = int(int(k * 100)/100) if int(int(k * 100)/100) == int(k * 100)/100 else int(k * 100)/100
                    l = int(int(l * 100)/100) if int(int(l * 100)/100) == int(l * 100)/100 else int(l * 100)/100
                    data_raw[(k,l)] = inf
        
        title = f"Feature Map ({self.combine_strategy})\nc: {'%.3f'% self.get_fm_coverage()}, s: {'%.3f'% self.get_fm_sparseness()}, fs: {'%.3f'% self.get_failure_sparsness()}"
        # print("filepath", filepath)
        # print(f"data_raw: {data_raw}")
        plot_map_of_elites(data=data_raw, 
                        filepath = filepath,
                        iterations = iterations,
                        x_axis_label = self.feature_names[0],
                        y_axis_label = self.feature_names[1],
                        min_value_cbar = self.max_fitness,
                        max_value_cbar = self.min_fitness,
                        title = title,
                        filename= filename,
                        title_suffix = title_suffix                      
        )

    def get_bin(self, feature_value: float, feature_id: int):
        id = feature_id
        return int ( (feature_value - self.bound_low[id]) / self.interval[id] ) 

    def get_fm_coverage(self):
        cells_total = 1
        for i in range(len(self.feature_names)):
            cells_total *= (self.bound_up[i] - self.bound_low[i]) / self.interval[i]
        return len(self.data) / cells_total
    
    def dist_manhatten(self, t1,t2):
        dist = 0
        for i in range(len(t1)):
            dist += abs(t1[i] - t2[i])
        return dist
    
    def get_failure_sparsness(self):
        import itertools
        # use manhatten distance to compute sparseness
        sparseness = 0
        if len(self.data.keys()) <= 1:
            return 0
        f_cells  = []
        for key,value in self.data.items():
            if value < self.failure_threshold:
                f_cells.append(key)

        # Generate 2-permutations
        two_perms = list(itertools.permutations(f_cells, 2))
        if len(two_perms) == 0:
            return 0
        
        # Print 2-permutations
        for perm in two_perms:
            sparseness += self.dist_manhatten(perm[0], perm[1])

        return sparseness/len(two_perms)
        
    def get_fm_sparseness(self):
        import itertools
        # use manhatten distance to compute sparseness
        sparseness = 0
        # Generate 2-permutations
        two_perms = list(itertools.permutations(self.data.keys(), 2))
        if len(two_perms) == 0:
            return 0
        
        # Print 2-permutations
        for perm in two_perms:
            sparseness += self.dist_manhatten(perm[0], perm[1])

        return sparseness/len(two_perms)
    
    def export_to_json(self, filename):
        with open(filename, 'w') as json_file:
            json.dump(self.to_dict(), json_file, indent=4,cls=NumpyEncoder)

    def to_dict(self):

        def get_n_data():
            n_all = 0
            for i,j in self.input_data_all.items():
                n_all += len(j)
            return n_all
        return {
            "feature_names": self.feature_names,
            "bound_low": self.bound_low,
            "bound_up": self.bound_up,
            "interval": self.interval,
            "min_fitness": self.min_fitness,
            "max_fitness": self.max_fitness,
            "data": {str(k): v for k, v in self.data.items()},
            "data_all": {str(k): v for k, v in self.data_all.items()},
            "data_clean": {str(k): v for k, v in self.data_clean.items()},
            "failure_threshold": self.failure_threshold,
            "input_data_all": {str(k): v for k, v in self.input_data_all.items()},
            "n_cell_fail" : sum([v < self.failure_threshold for v in self.data.values()]),
            "n_cell_covered" : len(self.data.keys()),
            "n_data" : get_n_data(),
            "combine_strategy": self.combine_strategy
        }

    @staticmethod
    def from_json(filename):
        with open(filename, 'r') as json_file:
            data = json.load(json_file)
        feature_map = FeatureMap(
            feature_names=data["feature_names"],
            bound_low=data["bound_low"],
            bound_up=data["bound_up"],
            interval=data["interval"],
            min_fitness=data["min_fitness"],
            max_fitness=data["max_fitness"],
            failure_threshold=data["failure_threshold"],
            combine_strategy=data["combine_strategy"]
        )
        feature_map.data = {FeatureMap.str_to_tuple(k): v for k, v in data["data"].items()}
        feature_map.data_all = {FeatureMap.str_to_tuple(k): v for k, v in data["data_all"].items()}
        feature_map.data_clean = {FeatureMap.str_to_tuple(k): v for k, v in data["data_clean"].items()}
        feature_map.input_data_all = {FeatureMap.str_to_tuple(k): v for k, v in data["input_data_all"].items()}
        return feature_map
    
    @staticmethod
    def str_to_tuple(s):
        if "(" in s:
            res = s.strip("()").split(", ")
        elif "[" in s:
            res = s.strip("[]").split(", ")
        else:
            pass
        return tuple([float(n) for n in res])
    
    @staticmethod
    def migrate_equal(fm1, fm2):
        # Migrate maps which contain equal tests
        # Ensure both feature maps are compatible
        # dont have to be necessary all the same tests
        
        assert fm1.feature_names == fm2.feature_names
        assert fm1.bound_low == fm2.bound_low
        assert fm1.bound_up == fm2.bound_up
        assert fm1.interval == fm2.interval
        assert fm1.min_fitness == fm2.min_fitness
        assert fm1.max_fitness == fm2.max_fitness
        assert fm1.failure_threshold == fm2.failure_threshold
        assert fm1.combine_strategy ==fm2.combine_strategy 
        # assert fm1.input_data_all == fm2.input_data_all

        migrated_fm = FeatureMap(
            feature_names=fm1.feature_names,
            bound_low=fm1.bound_low,
            bound_up=fm1.bound_up,
            interval=fm1.interval,
            min_fitness=fm1.min_fitness,
            max_fitness=fm1.max_fitness,
            failure_threshold=fm1.failure_threshold,
            combine_strategy=fm1.combine_strategy
            # assuming both have the same strategy
        )

        migrated_fm.data_all = {}
        migrated_fm.data = {}
        
        migrated_fm.data_clean = {key: max(fm1.data_clean.get(key, float('-inf')), fm2.data_clean.get(key, float('-inf')))
                 for key in set(fm1.data_clean) | set(fm2.data_clean)}
    
        migrated_fm.input_data_all = FeatureMap.concat_dict(fm1.input_data_all, fm2.input_data_all)

        for key in set(fm1.input_data_all) | set(fm2.input_data_all):
            migrated_fm.data_all[key] = []
            # print("key", key)
            tests1 = fm1.input_data_all.get(key, [])
            tests2 = fm2.input_data_all.get(key, [])
            
            # one of the maps doesnt have the cell
            if tests1 is None and tests2 is not None:
                migrated_fm.data_all[key] = fm2.data_all.get(key, [])
                migrated_fm.data[key] = fm2.data.get(key)
            if tests2 is None and tests1 is not None:
                migrated_fm.data_all[key] = fm1.data_all.get(key, [])
                migrated_fm.data[key] = fm1.data.get(key)
            else:
                # both have   
                # print(tests1 + tests2)
                filtered_list = list(map(list, set(map(tuple, tests1 + tests2))))
                migrated_fm.input_data_all[key] = filtered_list
                
                # print(filtered_list)
                # input()
                # get index for each test to retrieve fitness
                for test in filtered_list:
                    index1 = tests1.index(test) if test in tests1 else None
                    index2 = tests2.index(test) if test in tests2 else None
                    fitness1 = fm1.data_all.get(key, [])[index1] if index1 is not None else -10e3
                    fitness2 = fm2.data_all.get(key, [])[index2] if index2 is not None else -10e3
        
                    # if both have the test, select the one with the worse value (after optimization)
                    migrated_fm.data_all[key].append(
                            max(fitness1, fitness2))
                
                # print("fitness values are:", migrated_fm.data_all[key])
                migrated_fm.data[key] = min(migrated_fm.data_all[key]) # optimal approach

        return migrated_fm

    @staticmethod
    def migrate(fm1, fm2):
        assert fm1.feature_names == fm2.feature_names
        assert fm1.bound_low == fm2.bound_low
        assert fm1.bound_up == fm2.bound_up
        assert fm1.interval == fm2.interval
        assert fm1.min_fitness == fm2.min_fitness
        assert fm1.max_fitness == fm2.max_fitness
        assert fm1.failure_threshold == fm2.failure_threshold
        assert fm1.combine_strategy ==fm2.combine_strategy 

        merged_fm = FeatureMap(
            feature_names=fm1.feature_names,
            bound_low=fm1.bound_low,
            bound_up=fm1.bound_up,
            interval=fm1.interval,
            min_fitness=fm1.min_fitness,
            max_fitness=fm1.max_fitness,
            failure_threshold=fm1.failure_threshold,
            combine_strategy=fm1.combine_strategy  # assuming both have the same strategy
        )

        for key in set(fm1.data) | set(fm2.data):
            # Get the values for the key, defaulting to -inf if the key is missing
            value1 = fm1.data.get(key)
            value2 = fm2.data.get(key)
            if value1 is None and value2 is not None:
                merged_fm.data[key] = value2
            elif value1 is not None and value2 is None:
                merged_fm.data[key] = value1
            else:
                if value1 < value2:
                    merged_fm.data[key] = value1
                else:
                    merged_fm.data[key] = value2

        for key in set(fm1.data_clean) | set(fm2.data_clean):
            # Get the values for the key, defaulting to -inf if the key is missing
            value1 = fm1.data_clean.get(key)
            value2 = fm2.data_clean.get(key)
            if value1 is None and value2 is not None:
                merged_fm.data_clean[key] = value2
            elif value1 is not None and value2 is None:
                merged_fm.data_clean[key] = value1
            else:
                if value1 < value2:
                    merged_fm.data_clean[key] = value1
                else:
                    merged_fm.data_clean[key] = value2

        merged_fm.data_all = FeatureMap.concat_dict(fm1.data_all, fm2.data_all)#{**fm1.data_all, **fm2.data_all}
        merged_fm.input_data_all = FeatureMap.concat_dict(fm1.input_data_all, fm2.input_data_all)
        return merged_fm

    @staticmethod
    def concat_dict(data1,data2):
        concatenated_data = {}
        # Iterate through the keys of both dictionaries
        for key in set(data1) | set(data2):
            # Concatenate the lists for common keys
            concatenated_data[key] = data1.get(key, []) + data2.get(key, [])
        return concatenated_data
    
if __name__ == "__main__":
    feature_names = ["feature_1", "feature_2"]
    bound_up=[1,10]
    bound_low=[0,0]
    interval=[0.1,1]
    fmap: FeatureMap = FeatureMap(
            feature_names=feature_names,
            bound_up=bound_up,
            bound_low=bound_low,
            interval=interval,
            min_fitness=-3,
            max_fitness=0)
    fmap2: FeatureMap = FeatureMap(
            feature_names=feature_names,
            bound_up=bound_up,
            bound_low=bound_low,
            interval=interval,
            min_fitness=-3,
            max_fitness=0)

    values = {}
    values2 = {}

    N = 50
    # for i in range(0,N):
    #     features = ()
    #     for k,_ in enumerate(feature_names):
    #         v = random.randint(bound_low[k], bound_up[k])
    #         features += (v, )
    #     values[features] = random.random()
    values[(0.4,2)] = -0.5
    # values[(0.1,2)] = 0.9
    values[(0.3,2)] = -0.9
    values[(0.8,2)] = -0.9
    values[(0.1,8)] = -2.5
    # values[(0.18,2)] = -2.4
    values[(0.5,2)] = -2.9
    # values[(0.4,5)] = -2.9

        
    values2[(0.4,2)] = 0
    # values[(0.1,2)] = 0.9
    # values2[(0.5,2)] = -1.9
    values2[(0.8,2)] = -1.9
    values2[(0.1,8)] = -2.0
    #values2[(0.2,2)] = -2.0
    values2[(0.5,2)] = -2.9
    # values[(0.001,0)] = 0.1
    values2[(0.3,2)] = -0.9

    inputs = [[random.randint(-180,180) for i in range(5)] for i in range(2*len(values.keys()))]
    inputs2 = inputs

    # inputs2 = [[random.randint(-180,180) for i in range(5)] for i in range(2*len(values2.keys()))]

    i = 0
    for features, fitness in values.items():
        fmap.update_map(features,fitness, input_values = inputs[i])
        fmap.update_map(features,fitness, input_values = inputs[i+1])
        i += 2

    i = 0
    for features, fitness in values2.items():
        fmap2.update_map(features,fitness, input_values = inputs2[i])
        fmap2.update_map(features,fitness, input_values = inputs2[i+1])
        i += 2

    # write    
    file_path = os.getcwd() + os.sep + "features" + os.sep + "output_fmap.json"
    with open(file_path, 'w') as json_file:
        json_file.write(fmap.data_to_json())

    file_path = os.getcwd() + os.sep + "features" + os.sep + "output_input_fmap.json"
    with open(file_path, 'w') as json_file:
        json_file.write(fmap.input_data_all_to_json())

    fmap.plot_map(filename="fmap1")
    fmap.export_to_json(
            os.path.join(os.getcwd(), "features","fmap.json")
    )

    fmap2.plot_map(filename="fmap2")
    fmap2.export_to_json(
        os.getcwd() + os.sep + "features" + os.sep + "fmap2.json"
    )

    #################

    # map_migrated = FeatureMap.migrate(map,map2)
    # map_migrated.plot_map(filename="map_migrated")

    # map_migrated.export_to_json(
    #     os.getcwd() + os.sep + "features" + os.sep + "fmap_migrated.json"
    # )

    # print(f"Fmap coverage: {map.get_fm_coverage()}")
    # print(f"Failure sparseness: {map.get_failure_sparsness()}")
    # print(f"Fmap sparseness: {map.get_fm_sparseness()}")

    ######################
    
    # map_migrated_equal = FeatureMap.migrate_equal(fmap,fmap2)
    # map_migrated_equal.plot_map(filename="map_migrated_eq")

    # map_migrated_equal.export_to_json(
    #     os.getcwd() + os.sep + "features" + os.sep + "fmap_migrated_eq.json"
    # )
    
    ########################## real example
    fmap = FeatureMap.from_json(r"C:\Users\Lev\Documents\testing\Multi-Simulation\results\analysis\analysis_5-runs_08-09-2024_19-42-35_20-gen_20-pop\run_4\sim_1\VARSEG_Donkey_A10_-180-180_XTE_DIST_gen20_pop20_seed684\NSGAII-D\myfolder\backup\resumed\feature_map\gen_10\fmap.json")
    fmap2 = FeatureMap.from_json(r"C:\Users\Lev\Documents\testing\Multi-Simulation\results\analysis\analysis_5-runs_08-09-2024_19-42-35_20-gen_20-pop\run_4\sim_1\VARSEG_Donkey_A10_-180-180_XTE_DIST_gen20_pop20_seed684\NSGAII-D\myfolder\backup\resumed\validation_quarter_all_udacity\validation_udacity_fmap.json")
    
    map_migrated_equal = FeatureMap.migrate_equal(fmap,fmap2)
    map_migrated_equal.plot_map(filename="map_migrated_eq_exps")

    map_migrated_equal.export_to_json(
        os.getcwd() + os.sep + "features" + os.sep + "fmap_migrated_eq_exps.json"
    )