"""Use and compare a zero forcing receiver and a MMSE receiver on data transmitted through a 4x4 channel.

This script is for the Principles of Wireless Communications Lab Part a.
"""
from __future__ import print_function, division

import matplotlib.pyplot as plt
import numpy as np

from util import signal_util, receivers

# Read transmitted signals.
tx1 = signal_util.load_complex_float32('data/tx1.dat')
tx2 = signal_util.load_complex_float32('data/tx2.dat')

signals = [(tx1, 'Transmitted antenna 1'), (tx2, 'Transmitted antenna 2')]
signal_util.make_subplots(signals)

# Read received signals.
rx1 = signal_util.load_complex_float32('data/rx1.dat')
rx2 = signal_util.load_complex_float32('data/rx2.dat')

signals = [(rx1, 'Received antenna 1'), (rx2, 'Received antenna 2')]
signal_util.make_subplots(signals)

# Grab slices of the signal used in the receiver implementations.
tx_header1, tx_header2 = signal_util.get_headers_tx(tx1, tx2)
tx_data1, tx_data2 = signal_util.get_data_tx(tx1, tx2)

rx_header1, rx_header2, zeros1, zeros2, data1, data2 = signal_util.get_slices_rx(rx1, rx2)

# Calculate the SNR of the received signal.
noise_rx1 = rx1[:3 * signal_util.ZERO_SAMPLES // 4]
noise_rx2 = rx2[:3 * signal_util.ZERO_SAMPLES // 4]

snr = signal_util.calculate_snr(np.vstack((noise_rx1, noise_rx2)), np.vstack((data1, data2)))
print(np.average(snr), 'Average before recovery')

# Decode the transmitted data into a bit sequence for calculating bit-error
# rate.
symbols1_tx = signal_util.decode_bpsk(tx_data1)
symbols2_tx = signal_util.decode_bpsk(tx_data2)

# Estimate the channels between antennas.
H = receivers.estimate_channel(rx_header1, rx_header2, zeros1, zeros2,
        tx_header1, tx_header2)

# Recover signals with zero-forcing receiver.
W_zf = receivers.calculate_weights_zero_forcing(H)

x1_est, x2_est = receivers.recover_signals(rx1, rx2, W_zf)

# Plot the recovered signals.
estimates = [x1_est, x2_est]
estimate_titles = ['x1_estimate ZF', 'x2_estimate ZF']
signals = [(x, y) for x, y in zip(estimates, estimate_titles)]
signal_util.make_subplots(signals)

# Calculate the bit-error rate of recovered signals.
_, _, noise_1, noise_2, data1_zf, data2_zf = signal_util.get_slices_rx(x1_est, x2_est)
symbols1_zf = signal_util.decode_bpsk(data1_zf)
symbols2_zf = signal_util.decode_bpsk(data2_zf)

error1_zf = signal_util.calculate_error_rate(symbols1_tx, symbols1_zf)
error2_zf = signal_util.calculate_error_rate(symbols2_tx, symbols2_zf)

# Calculate the SNR of the recovered signals.
snr = signal_util.calculate_snr(np.vstack((noise_1, noise_2)), np.vstack((data1_zf, data2_zf)))
print(np.average(snr))

print("ZF error in 1: {}, in 2: {}".format(error1_zf, error2_zf))

# Recover signals with MMSE receiver.
tx_power = 1
sigma = signal_util.estimate_noise_var(rx1)
W_mmse = receivers.calculate_weights_mmse(tx_power, sigma, H)

x1_est, x2_est = receivers.recover_signals(rx1, rx2, W_mmse)

# Plot the recovered signals.
estimates = [x1_est, x2_est]
estimate_titles = ['x1_estimate MMSE', 'x2_estimate MMSE']
signals = [(x, y) for x, y in zip(estimates, estimate_titles)]
signal_util.make_subplots(signals)

# Calculate the bit-error rate of the recovered signals.
_, _, noise_1, noise_2, data1_mmse, data2_mmse = signal_util.get_slices_rx(x1_est, x2_est)
symbols1_mmse = signal_util.decode_bpsk(data1_mmse)
symbols2_mmse = signal_util.decode_bpsk(data2_mmse)

error1_mmse = signal_util.calculate_error_rate(symbols1_tx, symbols1_mmse)
error2_mmse = signal_util.calculate_error_rate(symbols2_tx, symbols2_mmse)

# Calculate the SNR of the recovered signals.
snr = signal_util.calculate_snr(np.vstack((noise_1, noise_2)), np.vstack((data1_mmse, data2_mmse)))
print(np.average(snr))

print("MMSE error in 1: {}, in 2: {}".format(error1_mmse, error2_mmse))
