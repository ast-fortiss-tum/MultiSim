import numpy as np
from pymoo.core.population import Population
from config import DUPLICATE_COMP_PRECISION

def duplicate_free_with_index(population,precision=DUPLICATE_COMP_PRECISION):
    inds = population.get("X")
    indeces = remove_duplicates(inds, precision)
    dup_free = [population[i] for i in indeces]
    return Population(individuals = dup_free), indeces

def duplicate_free(population,precision=DUPLICATE_COMP_PRECISION):
    inds = population.get("X")
    dup_free = [population[i] for i in remove_duplicates(inds, precision)]
    return Population(individuals = dup_free)

''' returns indices of elements of array M after elimination of duplicates '''
def remove_duplicates(M,precision=DUPLICATE_COMP_PRECISION):
    res = []
    
    size = M.shape[0]
    if len(M) == 1:
        return [0]
    elif len(M) == 0:
        return []

    # round to x.th digit
    D = []
    for i in range(0,size):
        rounded = np.array([np.round(v,precision) for v in M[i]])
        D.append(rounded)

    D = np.asarray(D)
    I = np.lexsort([D[:, i] for i in reversed(range(0, M.shape[1]))])
    S = D[I, :]

    i = 0
    # filter duplicates
    while i < size - 1:
        res.append(I[i])
        while np.all(S[i, :] == S[i + 1, :]):
            if i == size - 2:
                return res
            else:
                i += 1
        i = i + 1 
        if i == (size - 1):
            res.append(I[i])
            return res    
    return res


# M0 = np.asarray([[1.00001,2],[1.00,2],[1.00081,2],[1.2,3]])
# assert ( remove_duplicates(M0) == [1,0,2,3] )

# M1 = np.asarray([[1],[2],[2],[4]])
# assert ( remove_duplicates(M1) == [0,1,3] )

# M2 = np.asarray([[1],[2],[3],[4]])
# assert ( remove_duplicates(M2) == [0,1,2,3] )

# M3 = np.asarray([[1],[2],[3],[3]])
# assert ( remove_duplicates(M3) == [0,1,2] )

# M4 = np.asarray([[1.2,3]])
# assert ( remove_duplicates(M4) == [0] )

# M5 = np.asarray([[1.2,3], [1.2, 3]])
# assert ( remove_duplicates(M5) == [0] or remove_duplicates(M5) == [1] )

# M6 = np.asarray([])
# assert ( remove_duplicates(M6) == [] )