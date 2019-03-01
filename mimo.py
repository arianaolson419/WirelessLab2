from __future__ import print_function, division

import matplotlib.pyplot as plt
import numpy as np

from MIMOChannel import MIMOChannel4x4
from util import signal_util, receivers

# Generate matrices containing data and header bits.
data = signal_util.generate_symbols_mimo(signal_util.DATA_BITS, 10)
headers = signal_util.generate_symbols_mimo(signal_util.HEADER_BITS, 5)

# Generate the samples to transmit.
bpsk_data = signal_util.generate_data_mimo(data, signal_util.PULSE_SIZE)
bpsk_headers = signal_util.generate_data_mimo(headers, signal_util.PULSE_SIZE)

# Put it all together into a matrix of 4 complete signals. See doc comments on
# signal_util.create_tx_mimo for information on how these were generated.
tx_mimo = signal_util.create_tx_mimo(bpsk_headers, bpsk_data, signal_util.ZERO_SAMPLES)

signals = [(tx_mimo[i, :], 'tx {}'.format(i)) for i in range(4)]
signal_util.make_subplots(signals)

# Need to apply a gain before transmitting through the channel.
gain_amplitude = 200

# Send the signal through the channel.
rx_mimo = MIMOChannel4x4(gain_amplitude * tx_mimo)

signals = [(rx_mimo[i, :], '{} rx'.format(i)) for i in range(4)]
signal_util.make_subplots(signals)

# TODO: Consider using xcorr to find sections of data instead of doing it
# manually.

# Find the indices of the portions of the signals used to estimate the
# channels and the data portions of the signals.
header_samples = signal_util.PULSE_SIZE * signal_util.HEADER_BITS
data_samples = signal_util.PULSE_SIZE * signal_util.DATA_BITS

header1_start = signal_util.ZERO_SAMPLES
header1_end = signal_util.ZERO_SAMPLES + header_samples

header2_start = signal_util.ZERO_SAMPLES + header1_end
header2_end = header2_start + header_samples

header3_start = signal_util.ZERO_SAMPLES + header2_end
header3_end = header3_start + header_samples

header4_start = signal_util.ZERO_SAMPLES + header3_end
header4_end = header4_start + header_samples

data_start = header4_end + signal_util.ZERO_SAMPLES
data_end = data_start + data_samples

# Initialize matrix to hold signal slices in. A signal at sections[i, j, :]
# corresponds to the section of rx at antenna i + 1 where a header was sent
# from tx antenna j + 1.
sections = np.zeros((4, 4, header_samples - 10), dtype=np.complex128)

# The headers at the antenna they were intended for.
header11 = np.copy(rx_mimo[0, header1_start:header1_end - 10])
header22 = np.copy(rx_mimo[1, header2_start:header2_end - 10])
header33 = np.copy(rx_mimo[2, header3_start:header3_end - 10])
header44 = np.copy(rx_mimo[3, header4_start:header4_end - 10])

sections[0, 0, :] = header11
sections[1, 1, :] = header22
sections[2, 2, :] = header33
sections[3, 3, :] = header44

# The interference of header 1 in non-rx antenna 1 signals.
zeros21 = np.copy(rx_mimo[1, header1_start:header1_end - 10])
zeros31 = np.copy(rx_mimo[2, header1_start:header1_end - 10])
zeros41 = np.copy(rx_mimo[3, header1_start:header1_end - 10])

sections[1, 0, :] = zeros21
sections[2, 0, :] = zeros31
sections[3, 0, :] = zeros41

# The interference of header 2 in non-rx antenna 2 signals.
zeros12 = np.copy(rx_mimo[0, header2_start:header2_end - 10])
zeros32 = np.copy(rx_mimo[2, header2_start:header2_end - 10])
zeros42 = np.copy(rx_mimo[3, header2_start:header2_end - 10])

sections[0, 1, :] = zeros12
sections[2, 1, :] = zeros32
sections[3, 1, :] = zeros42

# The interference of header 3 in non-rx antenna 3 signals.
zeros13 = np.copy(rx_mimo[0, header3_start:header3_end - 10])
zeros23 = np.copy(rx_mimo[1, header3_start:header3_end - 10])
zeros43 = np.copy(rx_mimo[3, header3_start:header3_end - 10])

sections[0, 2, :] = zeros13
sections[1, 2, :] = zeros23
sections[3, 2, :] = zeros43

# The interference of header 4 in non-rx antenna 4 signals.
zeros14 = np.copy(rx_mimo[0, header4_start:header4_end - 10])
zeros24 = np.copy(rx_mimo[1, header4_start:header4_end - 10])
zeros34 = np.copy(rx_mimo[2, header4_start:header4_end - 10])

sections[0, 3, :] = zeros14
sections[1, 3, :] = zeros24
sections[2, 3, :] = zeros34

# Estimate the channel from these signal slices.
H_zf = receivers.estimate_channel_mimo(sections, bpsk_headers)
print(H_zf)

# Calculate the ZF weight matrix
W_zf = receivers.calculate_weights_zero_forcing(H_zf)

# Use ZF to estimate the signals.
x_est = receivers.recover_signals_mimo(rx_mimo, W_zf)

# Plot the recovered signals.
signals = [(x_est[i, :], 'x_est {}'.format(i)) for i in range(4)]
signal_util.make_subplots(signals)

# Visualize the slices of the signals used to isolate the headers.
tmp = np.zeros(rx_mimo.shape, dtype=rx_mimo.dtype)
tmp[0, header1_start:header1_end - 10] = header11
tmp[1, header2_start:header2_end - 10] = header22
tmp[2, header3_start:header3_end - 10] = header33
tmp[3, header4_start:header4_end - 10] = header44

plt.subplot(4, 1, 1)
plt.plot(rx_mimo[0, :])
plt.plot(tmp[0, :])
plt.subplot(4, 1, 2)
plt.plot(rx_mimo[1, :])
plt.plot(tmp[1, :])
plt.subplot(4, 1, 3)
plt.plot(rx_mimo[2, :])
plt.plot(tmp[2, :])
plt.subplot(4, 1, 4)
plt.plot(rx_mimo[3, :])
plt.plot(tmp[3, :])
plt.show()

