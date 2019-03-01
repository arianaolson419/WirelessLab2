"""This file contains functions for loading, inspecting, and slicing signals
for use in Zero-Forcing and MMSE receivers.
"""
import numpy as np
import matplotlib.pyplot as plt

HEADER_BITS = 128
DATA_BITS = 1024
PULSE_SIZE = 40
ZERO_SAMPLES = 5000
TOTAL_SAMPLES = PULSE_SIZE * (HEADER_BITS * 2 + DATA_BITS) + ZERO_SAMPLES * 3

def load_complex_float32(path):
    """Loads a .dat file of 32 bit floating point values as a single complex
    signal.

    It is assumed that the .dat file being loaded encodes complex numbers as
    interleaved 32 bit floating point real and imaginary values. This function
    formats that data into a single complex array.

    Args:
        path (string): The relative path to the .dat file.

    Returns:
        y (complex 1D ndarray): The complex signal.
    """
    tmp = np.fromfile(path, dtype=np.float32)

    y = tmp[::2] + 1j * tmp[1::2]

    # Make these vectors read-only to avoid inadvertently changing them later.
    y.flags.writeable = False
    return y

def get_headers_tx(tx1, tx2):
    """Grab the 128 psuedo-random bits from the transmitted signals.

    The transmitted signals are each made up of the following parts:

    1. 5000 zero samples.
    2. 128 psuedo-random bits from tx1, encoded using BPSK and 40 sample long
        rectangular pulses. 128 zeros from tx2.
    3. 5000 zero samples.
    4. 128 psuedo-random bits from tx2, encoded using BPSK and 40 sample long
        rectangular pulses. 128 zeros from pulse 2.
    5. 5000 zero samples from both tx1 and tx2.
    6. 1024 data bits, encoded using BPSK and 40 sample long rectangular pulses from both tx1 and
        tx2.

    Args:
        tx1 (complex numpy array): The signal transmitted at antenna 1.
        tx2 (complex numpy array): The signal transmitted at antenna 2.

    Returns:
        header1, header2 (complex numpy arrays): The headers of tx1 and tx2, respectively.
    """
    header1_start = ZERO_SAMPLES
    header1_end = header1_start + (HEADER_BITS * PULSE_SIZE)
    header1 = np.copy(tx1[header1_start:header1_end])
    
    header2_start = header1_end + ZERO_SAMPLES
    header2_end = header2_start + (HEADER_BITS * PULSE_SIZE)
    header2 = np.copy(tx2[header2_start:header2_end])

    return header1, header2

def get_data_tx(tx1, tx2):
    """Extract the data portion of the received signals.
    
    Args:
        tx1 (complex 1D ndarray): The signal transmitted at antenna 1.
        tx2 (complex 1D ndarray): The signal transmitted at antenna 2.

    Returns:
        data1, data2

        data1 (complex 1D ndarray): The data portion of tx1.
        data2 (complex 1D ndarray): The data portion of tx2.
    """
    data_start = 3 * ZERO_SAMPLES + 2 * (HEADER_BITS * PULSE_SIZE)
    data_end = data_start + (DATA_BITS * PULSE_SIZE)

    data1 = np.copy(tx1[data_start:data_end])
    data2 = np.copy(tx2[data_start:data_end])

    return data1, data2

def get_slices_rx(rx1, rx2):
    """Grab the segments of the received signals corresponding to when the two headers were transmitted.

    A 128 bit pseudo-random header is transmitted from antenna 1, followed by a
    series of 5000 zero samples, then a 128 bit header is sent from antenna 2.
    This function grabs the segments of both received signals corresponding to
    the times at which the transmitted headers were received.

    Args:
        rx1 (complex numpy array): The signal transmitted at antenna 1.
        rx2 (complex numpy array): The signal transmitted at antenna 2.

    Returns:
        header1, header2, zeros1, zeros2, data1, data2

        header1, header2 (complex numpy arrays): The segments of rx1 and rx2
            corresponding to where there should be a header ideally.
        zero1, zero2 (complex numpy arrays): The segments of rx1 and rx2
            correspinding to where a header was transmitted and we should ideally
            expect zeros. 
        data1, data2 (complex numpy arrays): The data portions of rx1 and rx2.
    """
    rx1_thresh = np.sqrt(np.mean(np.square(np.abs(rx1))))

    # These values were found through inspection of the data. This is very
    # specific to the data and should not be used as a generalized function.
    header1_start = np.argmax(np.abs(rx1) > rx1_thresh) - 2
    header1_end = header1_start + (HEADER_BITS * PULSE_SIZE)

    header1 = np.copy(rx1[header1_start:header1_end])
    zeros2 = np.copy(rx2[header1_start:header1_end])

    header2_start = header1_end + ZERO_SAMPLES
    header2_end = header2_start + (HEADER_BITS * PULSE_SIZE)

    header2 = np.copy(rx2[header2_start:header2_end])
    zeros1 = np.copy(rx1[header2_start:header2_end])

    data1_start = header2_end + ZERO_SAMPLES
    data2_start = header2_end + ZERO_SAMPLES

    data1_end = data1_start + (DATA_BITS * PULSE_SIZE)
    data2_end = data2_start + (DATA_BITS * PULSE_SIZE)

    data1 = np.copy(rx1[data1_start:data1_end])
    data2 = np.copy(rx2[data2_start:data2_end])

    return header1, header2, zeros1, zeros2, data1, data2

def estimate_noise_var(rx):
    """Estimate the variance in the noise from a portion of the received signal that contains no data.

    Args:
        rx (complex 1D ndarray): A received signal.

    Returns:
        sigma (float): The estimated variance of the noise.
    """
    # Take a portion of the beginning of the signal as the noise.
    noise = rx[:3 * ZERO_SAMPLES // 4]
    return noise.var()

def make_subplots(signals):
    """Makes and displays subplots of given signals.

    Args:
        signals (list of tuples): A list of tuples where the first element is
            the signal data to plot, and the second element is the title of the
            subplot.
    """
    num_plots = len(signals)
    for i, signal in enumerate(signals):
        plt.subplot(num_plots, 1, i + 1)
        plt.plot(signal[0])
        plt.title(signal[1])
    plt.show()

def decode_bpsk(data):
    """Decode BPSK data into symbols.

    Args:
        data (complex 1D ndarray): A signal containing BPSK data.

    Returns:
        symbols (1D ndarray): An array of the decoded symbols.
    """
    symbols = np.zeros(data.shape[-1] // PULSE_SIZE)
    for i in range(0, data.shape[-1], PULSE_SIZE):
        samples = data[i:i + PULSE_SIZE]
        is_positive = (samples.real > 0).sum() > (PULSE_SIZE // 2)
        if is_positive:
            symbols[i // PULSE_SIZE] = 1
    return symbols

def calculate_error_rate(symbols_tx, symbols_rx):
    """Calculate the percent error rate of a received and decoded data signal.

    Args:
        symbols_tx (1D ndarray): An array of transmitted bits.
        symbols_rx (1D ndarray): An array of received bits.
    
    Returns:
        percent_error (float): The percent error of the decoded received signal.
    """
    assert symbols_tx.shape == symbols_rx.shape
    return 100 - (100 * (symbols_rx == symbols_tx).sum() / symbols_tx.shape)
