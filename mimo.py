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

#plt.close('all')

# Put it all together into a matrix of 4 complete signals. See doc comments on
# signal_util.create_tx_mimo for information on how these were generated.
tx_mimo = signal_util.create_tx_mimo(bpsk_headers, bpsk_data, signal_util.ZERO_SAMPLES)

signals = [(tx_mimo[i, :], 'tx {}'.format(i)) for i in range(4)]
signal_util.make_subplots(signals)

# Need to apply a gain before transmitting through the channel.
gain_amplitude = 200

# Send the signal through the channel.
rx_mimo = np.asarray(MIMOChannel4x4(gain_amplitude * np.mat(tx_mimo)))

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
sections = np.zeros((4, 4, header1_end - header1_start), dtype=np.complex128)

# The headers at the antenna they were intended for.
header11 = np.copy(rx_mimo[0, header1_start:header1_end])
header22 = np.copy(rx_mimo[1, header2_start:header2_end])
header33 = np.copy(rx_mimo[2, header3_start:header3_end])
header44 = np.copy(rx_mimo[3, header4_start:header4_end])

sections[0, 0, :] = header11
sections[1, 1, :] = header22
sections[2, 2, :] = header33
sections[3, 3, :] = header44

# The interference of header 1 in non-rx antenna 1 signals.
zeros21 = np.copy(rx_mimo[1, header1_start:header1_end])
zeros31 = np.copy(rx_mimo[2, header1_start:header1_end])
zeros41 = np.copy(rx_mimo[3, header1_start:header1_end])

sections[1, 0, :] = zeros21
sections[2, 0, :] = zeros31
sections[3, 0, :] = zeros41

# The interference of header 2 in non-rx antenna 2 signals.
zeros12 = np.copy(rx_mimo[0, header2_start:header2_end])
zeros32 = np.copy(rx_mimo[2, header2_start:header2_end])
zeros42 = np.copy(rx_mimo[3, header2_start:header2_end])

sections[0, 1, :] = zeros12
sections[2, 1, :] = zeros32
sections[3, 1, :] = zeros42

# The interference of header 3 in non-rx antenna 3 signals.
zeros13 = np.copy(rx_mimo[0, header3_start:header3_end])
zeros23 = np.copy(rx_mimo[1, header3_start:header3_end])
zeros43 = np.copy(rx_mimo[3, header3_start:header3_end])

sections[0, 2, :] = zeros13
sections[1, 2, :] = zeros23
sections[3, 2, :] = zeros43

# The interference of header 4 in non-rx antenna 4 signals.
zeros14 = np.copy(rx_mimo[0, header4_start:header4_end])
zeros24 = np.copy(rx_mimo[1, header4_start:header4_end])
zeros34 = np.copy(rx_mimo[2, header4_start:header4_end])

sections[0, 3, :] = zeros14
sections[1, 3, :] = zeros24
sections[2, 3, :] = zeros34

# Estimate the channel from these signal slices.
H = receivers.estimate_channel_mimo(sections, bpsk_headers)
print(H)

# Calculate the ZF weight matrix
W_zf = receivers.calculate_weights_zero_forcing(H)

# Use ZF to estimate the signals.
x_est = receivers.recover_signals_mimo(rx_mimo, W_zf)

# Plot the recovered signals.
signals = [(x_est[i, :], 'x_est {}'.format(i)) for i in range(4)]
signal_util.make_subplots(signals)

# Transmit and recover MIMO signals with known CSI.
U, S, tx_transform = receivers.preprocess_tx(tx_mimo * gain_amplitude, H) 
rx_mimo_csi = np.asarray(MIMOChannel4x4(np.mat(tx_transform) * gain_amplitude))
s_est = receivers.recover_signals_csi(rx_mimo_csi, U)

signals_tx = [(tx_transform[i, :], 'Tx Transformed {}'.format(i + 1)) for i in range(4)]
signals_rx = [(rx_mimo_csi[i, :], 'Rx {}'.format(i + 1)) for i in range(4)]
signals_s_est = [(s_est[i, :], 's_est {}'.format(i + 1)) for i in range(4)]

#signal_util.make_subplots(signals_tx)
#signal_util.make_subplots(signals_rx)
#signal_util.make_subplots(signals_s_est)

tmp = np.zeros(x_est.shape)
tmp[:, data_start:data_end] = x_est[:, data_start:data_end]
plt.plot(x_est[0, :])
plt.plot(tmp[0, :])
plt.show()

bits_x_est_1 = signal_util.decode_bpsk(x_est[0, data_start:data_end])
error_x_est_1 = signal_util.calculate_error_rate(data[0, :], bits_x_est_1.real)

bits_x_est_2 = signal_util.decode_bpsk(x_est[1, data_start:data_end])
error_x_est_2 = signal_util.calculate_error_rate(data[1, :], bits_x_est_2.real)

bits_x_est_3 = signal_util.decode_bpsk(x_est[2, data_start:data_end])
error_x_est_3 = signal_util.calculate_error_rate(data[2, :], bits_x_est_3.real)

bits_x_est_4 = signal_util.decode_bpsk(x_est[3, data_start:data_end])
error_x_est_4 = signal_util.calculate_error_rate(data[3, :], bits_x_est_4.real)

print('''error rate from ZF receiver.
        Signal 1: {},
        Signal 2: {},
        signal 3: {},
        Signal 4: {}'''.format(error_x_est_1, error_x_est_2, error_x_est_3,
            error_x_est_4))
bits_s_est_1 = signal_util.decode_bpsk(s_est[0, data_start:data_end])
error_s_est_1 = signal_util.calculate_error_rate(data[0, :], bits_s_est_1.real)

bits_s_est_2 = signal_util.decode_bpsk(s_est[1, data_start:data_end])
error_s_est_2 = signal_util.calculate_error_rate(data[1, :], bits_s_est_2.real)

bits_s_est_3 = signal_util.decode_bpsk(s_est[2, data_start:data_end])
error_s_est_3 = signal_util.calculate_error_rate(data[2, :], bits_s_est_3.real)

bits_s_est_4 = signal_util.decode_bpsk(s_est[3, data_start:data_end])
error_s_est_4 = signal_util.calculate_error_rate(data[3, :], bits_s_est_4.real)

print('''error rate from CSI receiver.
        Signal 1: {},
        Signal 2: {},
        Signal 3: {},
        Signal 4: {}'''.format(error_s_est_1, error_s_est_2, error_s_est_3,
            error_s_est_4))
