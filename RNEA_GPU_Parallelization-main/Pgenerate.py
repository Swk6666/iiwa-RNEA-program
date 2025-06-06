import numpy as np

def Pgenerate(Imat_arr):
    # Initialize P as a zero vector of length 70
    P = np.zeros(70)

    for ind in range(7):
        # Extract the necessary components from each Imat (now Imat_arr is a numpy array)
        Imat = Imat_arr[ind]

        # S_cn is the 3x3 portion of the last 3x3 submatrix, divided by the (4,4) element
        S_cn = Imat[0:3, 3:6] / Imat[3, 3]

        # l is a vector created from specific elements of Imat
        l = np.array([Imat[0, 0], Imat[0, 1], Imat[0, 2],
                      Imat[1, 1], Imat[1, 2], Imat[2, 2]])

        # Update the appropriate slice of P
        P[(ind * 10):(ind * 10 + 10)] = np.concatenate(([Imat[3, 3]],
                                                        Imat[3, 3] * np.array([-S_cn[1, 2], S_cn[0, 2], -S_cn[0, 1]]),
                                                        l))

    return P
