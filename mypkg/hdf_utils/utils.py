import numpy as np

def gen_lam_seq(low, up, nlam):
    """
    Generate a sequence of lambda values.

    Parameters:
    low (float): The lower bound of the sequence.
    up (float): The upper bound of the sequence.
    nlam (int): The number of lambda values to generate.

    Returns:
    numpy.ndarray: A sequence of lambda values.
    """
    dlts =  np.linspace(-5, np.log(up-low), nlam-1)
    lam_seq = low + np.exp(dlts)
    lam_seq = np.concatenate([[low], lam_seq])
    return lam_seq