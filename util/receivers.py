"""This file contains functions to implement the receivers in both Part a and
Part b of the Principles of Wireless Communications Lab 2
"""

from __future__ import print_function, division
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
    tx1_header[tx1_header == 0] = 1e-12
    tx2_header[tx2_header == 0] = 1e-12
    
    H = np.zeros((2, 2), dtype=np.complex64)
    H[0][0] = np.mean(header11 / tx1_header)
    H[0][1] = np.mean(header12 / tx2_header)
    H[1][0] = np.mean(header21 / tx1_header)
    H[1][1] = np.mean(header22 / tx2_header)

    return H

def estimate_channel_mimo(rx_sections, tx_headers):
    """Estimate the channel using the headers sent from each antenna.

    Channel estimation is the same for both the zero-forcing and MMSE
    receivers. This function uses copies of the arrays given to it because any
    0s are replaced by small numbers ot avoid division errors.

    Args:
        rx_sections (complex (4, 4, header_bits) ndarray): A matrix of portions
            of the signal with the first two indices corresponding to the
            antenna the signal was received at and the antenna the header sent
            suring this time was transmitted from.
        tx_headers (ndarray of shape (4, header_bits)): The known transmitted
            headers.

    Returns:
        H (complex (4, 4) ndarray): A matrix of channel estimations.
    """
    # Replace 0s in denominators to avoid division errors.
    tx_headers[tx_headers == 0] = 1e-12

    H = np.zeros((4, 4), dtype=np.complex64)
    for i in range(4):
        for j in range(4):
            H[i, j] = np.mean(rx_sections[i, j, :] / tx_headers[i, :rx_sections.shape[-1]])

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

def recover_signals_mimo(rx, W):
    """Use a weight matrix to recover MIMO signals.
    Args:
        rx (complex (4, num_samples) ndarray): The received MIMO signals with
            each row as the received signal at one antenna.
        W (complex (4, 4) ndarray): A matrix of weights to apply to the signal.

    Returns:
        x_est (complex (4, num_samples) ndarray): The recovered MIMO signals in
            the same format as rx.
    """
    return np.matmul(W, rx)

def preprocess_tx(tx, H):
    """Transform the transmitted data with known channel information.

    Args:
        tx (complex (4, n) ndarray): The n sample long signals to transmit.
        H (complex (4, 4) ndarray): A matrix of channel estimations.

    Returns:
        U (complex (4, 4) ndarray): The unitary matrix U resulting from
            performing SVD on the channel estimations.
        S (complex (4, 4) ndarray): The diagonal matrix S representing the
            singular values of the SVD.
        tx_transform: The signal to transmit that's been transformed using CSI.
    """
    U, S, Vh = np.linalg.svd(H)
    tx_transform = np.matmul(np.transpose(np.conjugate(Vh)), tx)
    return U, S, tx_transform

def recover_signals_csi(rx, U):
    """Recover signals transformed with known CSI.

    Args:
        rx (complex (4, n) ndarray: The n sample long received signals.
        U (complex (4, 4) ndarray: The unitary matrix U from the SVD of the channel estimates.

    Returns:
        s_est: The recovered transmitted symbols.
    """
    s_est = np.matmul(np.transpose(np.conjugate(U)), rx)
    return s_est
