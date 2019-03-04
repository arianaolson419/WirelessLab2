"""This file contains functions for a creating and analyzing signals for both
Part a and Part b of the Principles of Wireless Communications Lab.
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
            corresponding to where a header was transmitted and we should ideally
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

def generate_symbols_mimo(num_symbols, seed):
    """Generate 4 random data sequences to transmit over the MIMO channel.

    Args: 
        num_symbols (int): The number of symbols of data to generate.
        seed (int): The seed for the random number generator.

    Returns:
        symbols (ndarray of shape (4, num_symbols)): An array consisting
        of +1 and -1 values.
    """
    np.random.seed(seed)
    symbols = np.sign(np.random.randn(4, num_symbols))
    
    return symbols

def generate_data_mimo(symbols, symbol_period):
    """Modulate a series of symbols into BPSK data.

    Args:
        symbols (numpy array of shape (4, num_symbols)): An array of 1s and -1s
            representing bits to transmit.
        symbol_period (int): The number of samples in a single pulse.
    
    Returns:
        bpsk_data (ndarray of shape (4, num_symbols * symbol_period)): An array
            of 4 bpsk data signals to transmit.
    """
    pulse = np.ones(symbol_period)
    bpsk_list = []

    for i in range(4):
        x = np.zeros(symbol_period * symbols.shape[-1]-symbol_period+1)
        x[::symbol_period] = symbols[0, :]
        tmp = np.convolve(x, pulse)
        bpsk_list.append(tmp)

    bpsk_data = np.vstack(bpsk_list)
    return bpsk_data

def create_tx_mimo(headers, data, num_zeros):
    """Creates 4 MIMO signals for transmitting through the channel.

    Each of the signals is constructed in the following way:
    1. A section of zero samples.
    2. A section of  psuedo-random bits from tx1, encoded using BPSK
    3. A section of zero samples.
    2. A section of  psuedo-random bits from tx2, encoded using BPSK
    3. A section of zero samples.
    2. A section of  psuedo-random bits from tx3, encoded using BPSK
    3. A section of zero samples.
    2. A section of  psuedo-random bits from tx4, encoded using BPSK
    5. A section of zero samples from all signals.
    6. A section of data bits, sent from all antennas.
    
    Args:
        headers (ndarray of shape (4, num_header_samples)): BPSK modulated headers to prepend to the data.
        data (ndarray of shape (4, num_data_samples)): BPSK modulated data to transmit.
        num_zeros(int): The number of zeros to use for padding.

    Returns:
        tx_mimo (ndarray of shape (4, total samples): The BPSK signals to transmit with MIMO over the channel.
    """
    header_samples = headers.shape[-1]
    data_samples = data.shape[-1]
    samples_per_signal = header_samples * 4 + data_samples + num_zeros * 5
    tx_mimo = np.zeros((4, samples_per_signal))
    
    header1_start = num_zeros
    header1_end = num_zeros + header_samples

    header2_start = num_zeros + header1_end
    header2_end = header2_start + header_samples
    
    header3_start = num_zeros + header2_end
    header3_end = header3_start + header_samples

    header4_start = num_zeros + header3_end
    header4_end = header4_start + header_samples

    data_start = header4_end + num_zeros
    data_end = data_start + data_samples

    tx_mimo[0, header1_start:header1_end] = headers[0, :]
    tx_mimo[1, header2_start:header2_end] = headers[1, :]
    tx_mimo[2, header3_start:header3_end] = headers[2, :]
    tx_mimo[3, header4_start:header4_end] = headers[3, :]

    tx_mimo[:, data_start:data_end] = data

    return tx_mimo

def rms(signals):
    """Calculate the RMS values of signals.

    Args:
        signals (complex 2D ndarray): A matrix with rows containing signals.

    Returns:
        rms (real 1D ndarray): A vector with entries corresponding to the rms
            of each row of signals.
    """
    rms = np.zeros(signals.shape[0])
    for i in range(signals.shape[0]):
        rms[i] = np.sqrt(np.mean(np.square(np.abs(signals[i, :]))))

    return rms

def linear_to_db(linear_signal):
    """Convert a signal from linear to dB.

    In this case, dB is in reference to 1.0.

    Args:
        linear_signal (real 2D ndarray): A matrix with rows containing the linear signals.

    Returns:
        db_signal (real 2D ndarray): A matrix with rows containing the signals
            with samples mesaured in dB.
    """
    reference = 1.0
    return 20 * np.log10(linear_signal)

def calculate_snr(noise_signal, data_signal):
    """Calculate the signal to noise ratio from the given signals.

    Args:
        noise_signal (complex 2D ndarray): A matrix with rows containing
            portions of a signal with just noise. The samples of the signal are
            linear amplitudes.
        data_signal (complex 2D ndarray): A matrix with rows containing
            portions of a signal with data and additive noise. The samples of
            the signal are linear amplitudes.

    Returns:
        snr (float): The signal to noise ratio between noise_signal and data_signal, in dB.
    """
    rms_noise = rms(noise_signal)
    rms_data = rms(data_signal)

    noise_db = linear_to_db(rms_noise)
    data_db = linear_to_db(rms_data)

    snr = data_db - noise_db
    return snr
