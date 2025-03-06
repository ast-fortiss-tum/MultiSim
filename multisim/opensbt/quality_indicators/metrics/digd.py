import numpy as np
from opensbt.utils.duplicates import remove_duplicates

def digd(real, solution_x, lb = None, ub = None):
    X = solution_x
    C = real
    digd = 0

    # make C,X duplicate free
    C = np.array([C[i] for i in remove_duplicates(C)])    
    X = np.array([X[i] for i in remove_duplicates(X)])

    C[:,0]= np.divide(C[:,0],np.linalg.norm(ub[0]- lb[0]))
    C[:,1]= np.divide(C[:,1],np.linalg.norm(ub[1]- lb[1]))

    # print(f"x0 normalizer: {np.linalg.norm(ub[0]- lb[0])}")
    # print(f"x1 normalizer: {np.linalg.norm(ub[1]- lb[1])}")

    # TODO normalize
    for i in range(len(C)):
        # if critical solution found continue
        if any((X[:]==C[i]).all(1)):#
            continue
        else:
            dist_c = []
            # find the closest solution if critical solution not found
            for j in range(len(X)):
                v = X[j,:] - C[i]
                dist_c.append(np.linalg.norm(v))
            digd += np.min(dist_c)

    if len(C) != 0:
        digd = digd / len(C)
    # normalize
    # d = np.linalg.norm(lb - ub)
    # digd /= d
    return digd

# test
if __name__ == "__main__":        
    real = np.array([[0,1],[0,2],[1,1],[2,1],[3,1]])
    solution_x = np.array([[0,2],[1,1],[2,1],[5,1]])
    ub = np.array([0,2])
    lb = np.array([5,1])

    print(digd(real,solution_x,ub,lb))


