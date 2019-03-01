"""This file contains functions to implement a Zero-Forcing receiver and an MMSE receiver.
"""

import numpy as np
import matplotlib.pyplot as plt

def estimate_channel(rx1_header, rx2_header, zeros1, zeros2, tx1_header, tx2_header):
    """Estimate the channel using the headers sent from each antenna.

    Channel estimation is the same for both the zero-forcing and MMSE
    receivers. This function uses copies of the arrays given to it because any
    0s are replaced by small numbers ot avoid division errors.

    Args:
        rx1_header (complex 1D ndarray): The portion of the signal received at
            rx antenna 1 corresponding to the header transmitted at tx antenna
            1.
        rx2_header (complex 1D ndarray): The portion of the signal received at
            rx antenna 2 corresponding to the header transmitted at tx antenna
            2.
        zeros1 (complex 1D ndarray): The portion of the signal received at rx
            antenna 1 corresponding the the header transmitted at tx antenna
            2.
        zeros2 (complex 1D ndarray): The portion of the signal received at rx
            antenna 2 corresponding the the header transmitted at tx antenna
            1.
        tx1_header (complex 1D ndarray): The header transmitted from tx antenna
            1.
        tx2_header (complex 1D ndarray): The header transmitted from tx antenna
            2.

    Returns:
        H (complex (2, 2) ndarray): A matrix of channel estimations.
    """
    header11 = np.copy(rx1_header)
    header21 = np.copy(zeros2)
    header12 = np.copy(zeros1)
    header22 = np.copy(rx2_header)

    # Replace 0s in denominators to avoid division errors.
    header11[header11 == 0] = 1e-12
    header21[header21 == 0] = 1e-12
    header12[header12 == 0] = 1e-12
    header22[header22 == 0] = 1e-12
    
    H = np.zeros((2, 2), dtype=np.complex64)
    H[0][0] = np.mean(tx1_header / header11)
    H[0][1] = np.mean(tx2_header / header12)
    H[1][0] = np.mean(tx1_header / header21)
    H[1][1] = np.mean(tx2_header / header22)

    return H

def calculate_weights_zero_forcing(H):
    """Calculates the weight matrix for the zero-forcing receiver.

    Args:
        H (complex (2, 2) ndarray): A matrix of channel estimations.

    Returns:
        W (complex (2, 2) ndarray): A matrix of weights.
    """
    W = np.linalg.inv(H)
    return W

def calculate_weights_mmse(tx_power, sigma, H):
    """Calculates the weight vectors for the MMSE receiver.

    Args:
        tx_power (float): The power of the transmitted signal.
        H (complex (2, 2) ndarray): A matrix of channel estimations.
        rx1 (complex 1D ndarray): The signal received at rx antenna 1.
        rx2 (complex 1D ndarray): The signal received at rx antenna 2.

    Returns: 
        w1 (complex 1D ndarray): The weight vector for recovering rx1.
        w2 (complex 1D ndarray): The weight vector for recovering rx2.
    """
    # Calculate R.
    h1 = H[0, :]
    h2 = H[1, :]

    R = tx_power * h1 * np.transpose(np.conjugate(h1)) + tx_power * h2 \
        * np.transpose(np.conjugate(h2)) + sigma * np.eye(2)

    # Calculate the weight matrix
    W = tx_power * np.linalg.inv(R) * H

    return W

def recover_signals(rx1, rx2, W):
    """Estimates the sent signals using the weight matrix.

    Args:
        rx1 (complex 1D ndarray): The signal received at rx antenna 1.
        rx2 (complex 1D ndarray): The signal received at rx antenna 2.
        W (complex (2, 2) ndarray): A matrix of weights.

    Returns:
        x1_est (complex 1D ndarray): The estimated signal transmitted from
            tx antenna 1.
        x2_est (complex 1D ndarray): The estimated signal transmitted from
            tx antenna 2.
    """
    y1 = np.copy(rx1)
    y2 = np.copy(rx2)

    ys = np.vstack((y1, y2))
    x_est = np.matmul(W, ys)

    x1_est = np.squeeze(x_est[0, :])
    x2_est = np.squeeze(x_est[1, :])

    return x1_est, x2_est
