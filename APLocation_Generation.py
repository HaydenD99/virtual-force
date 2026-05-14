import numpy as np
import numpy.matlib

def RandomAPLocations(L, squareLength):
    # Number of APs per dimension
    nbrAPsPerDim = int(np.sqrt(L))

    # Distance between APs in vertical/horizontal direction
    interAPDistance = squareLength / nbrAPsPerDim

    # Deploy APs on the grid
    locationsGridHorizontal = np.matlib.repmat(np.linspace(interAPDistance / 2, squareLength - interAPDistance / 2, nbrAPsPerDim), nbrAPsPerDim, 1)
    locationsGridVertical = np.transpose(locationsGridHorizontal)

    APpositions = locationsGridHorizontal.reshape((L, 1), order='F') + 1j * locationsGridVertical.reshape((L, 1), order='F')

    # Randomness introduced in AP locations
    sensitivity = 0.5  # 0 means no movement, 1 means max distance is interAPDistance
    min_distance = interAPDistance * (1 - sensitivity)
    assert interAPDistance >= min_distance

    max_movement = (interAPDistance - min_distance) / 2
    perturbation = np.random.uniform(
        low=-max_movement,
        high=max_movement,
        size=(len(APpositions), 2))

    perturbationXY = perturbation[:, 0].reshape((L, 1), order='F') + 1j * perturbation[:, 1].reshape((L, 1), order='F')
    APpositions += perturbationXY

    return APpositions

# Test the function
# print(RandomAPLocations(16, 1000))
