# modem.py
import numpy as np
import math
import wave
import io
import struct
from typing import Tuple
import scipy.signal as signal
import numpy as np
import math
import wave
import io
import struct
from typing import Tuple, Optional
import scipy.signal as signal
from scipy.fft import fft, ifft
from config import CONFIG

SAMPLE_RATE = 96000


class AdvancedModem:
    def __init__(self):
        self.sample_rate = CONFIG.get('modem.sample_rate', 96000)
        self.adaptive_equalization = CONFIG.get('modem.adaptive_equalization', True)
        self.noise_reduction = CONFIG.get('modem.noise_reduction', True)

    def _adaptive_filter(self, data: np.ndarray, cutoff_freq: float) -> np.ndarray:
        if not self.noise_reduction:
            return data

        nyquist = self.sample_rate / 2
        normal_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(6, normal_cutoff, btype='low')
        return signal.filtfilt(b, a, data)

    def _calculate_snr(self, signal_data: np.ndarray) -> float:
        if len(signal_data) < 1000:
            return 30.0

        b, a = signal.butter(3, 0.1, btype='high')
        noise_estimate = signal.filtfilt(b, a, signal_data)

        signal_power = np.mean(signal_data ** 2)
        noise_power = np.mean(noise_estimate ** 2)

        return 10 * np.log10(signal_power / (noise_power + 1e-10)) if noise_power > 0 else 50.0

    def _adaptive_gain_control(self, data: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(data ** 2))
        if rms > 0:
            target_rms = 0.3
            gain = target_rms / rms
            return data * min(gain, 5.0)
        return data


def fsk_modulate(data_bytes: bytes, baud=1200, mark_freq=1200.0,
                           space_freq=2200.0, samp_rate=96000) -> np.ndarray:
    modem = AdvancedModem()

    print(f"🎵 Modulando FSK Otimizada: {len(data_bytes)} bytes, baud={baud}")

    if baud >= 9600:
        mark_freq = 8000.0
        space_freq = 16000.0

    bit_dur = 1.0 / baud
    samples_per_bit = int(round(samp_rate * bit_dur))

    t = np.arange(samples_per_bit) / samp_rate
    mark_wave = np.sin(2 * np.pi * mark_freq * t)
    space_wave = np.sin(2 * np.pi * space_freq * t)
    sync_wave = np.sin(2 * np.pi * space_freq * t)

    bits = []
    for byte in data_bytes:
        bits.extend([int(b) for b in f'{byte:08b}'])

    total_samples = samples_per_bit * (len(bits) + 2)
    out = np.zeros(total_samples, dtype=np.float32)

    out[:samples_per_bit] = sync_wave
    ptr = samples_per_bit

    for bit in bits:
        out[ptr:ptr + samples_per_bit] = mark_wave if bit else space_wave
        ptr += samples_per_bit

    out[ptr:ptr + samples_per_bit] = sync_wave

    out = modem._adaptive_gain_control(out)

    m = np.max(np.abs(out))
    if m > 0:
        out = out / m * 0.8

    return out


def fsk_demodulate(samples: np.ndarray, baud=1200, mark_freq=1200.0,
                            space_freq=2200.0, samp_rate=96000) -> bytes:
    modem = AdvancedModem()

    print(f"🔍 Demodulando FSK Avançada: {len(samples)} amostras")

    if baud >= 9600:
        mark_freq = 8000.0
        space_freq = 16000.0

    snr = modem._calculate_snr(samples)
    cutoff = min(space_freq + 2000, samp_rate / 2 * 0.9)

    if snr < 20:
        cutoff = space_freq + 1000

    filtered = modem._adaptive_filter(samples, cutoff)

    samples_per_bit = int(round(samp_rate / baud))

    def optimized_goertzel(chunk, target_freq):
        w = 2.0 * math.pi * target_freq / samp_rate
        cos_w = math.cos(w)
        coeff = 2.0 * cos_w

        s_prev, s_prev2 = 0.0, 0.0
        for x in chunk:
            s = x + coeff * s_prev - s_prev2
            s_prev2, s_prev = s_prev, s
        return s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2

    bits = []
    for i in range(0, len(filtered) - samples_per_bit + 1, samples_per_bit):
        chunk = filtered[i:i + samples_per_bit]
        p_mark = optimized_goertzel(chunk, mark_freq)
        p_space = optimized_goertzel(chunk, space_freq)
        bits.append(1 if p_mark > p_space else 0)

    out = bytearray()
    for i in range(0, len(bits) - 7, 8):
        byte = sum((bit << (7 - j)) for j, bit in enumerate(bits[i:i+8]))
        out.append(byte)

    return bytes(out)


def bpsk_modulate(data_bytes: bytes, baud=1200, carrier=3000.0, samp_rate=96000) -> np.ndarray:
    bit_dur = 1.0 / baud
    samples_per_bit = int(round(samp_rate * bit_dur))

    t = np.arange(samples_per_bit) / samp_rate
    carrier_wave = np.sin(2 * np.pi * carrier * t)

    bits = [int(b) for byte in data_bytes for b in f'{byte:08b}']

    out = np.concatenate([carrier_wave if bit else -carrier_wave for bit in bits])
    m = np.max(np.abs(out))
    if m > 0:
        out = out / m * 0.8
    return out.astype(np.float32)


def bpsk_demodulate(samples: np.ndarray, baud=1200, carrier=3000.0, samp_rate=96000) -> bytes:
    samples_per_bit = int(round(samp_rate / baud))

    t = np.arange(samples_per_bit) / samp_rate
    ref = np.sin(2 * np.pi * carrier * t)

    bits = []
    for i in range(0, len(samples) - samples_per_bit + 1, samples_per_bit):
        chunk = samples[i:i + samples_per_bit]
        corr = np.sum(chunk * ref)
        bits.append(1 if corr > 0 else 0)

    out = bytearray()
    for i in range(0, len(bits) - 7, 8):
        byte = sum((bit << (7 - j)) for j, bit in enumerate(bits[i:i+8]))
        out.append(byte)

    return bytes(out)


def qpsk_modulate(data_bytes: bytes, baud=1200, carrier=3000.0, samp_rate=96000) -> np.ndarray:
    bits = ''.join(f'{byte:08b}' for byte in data_bytes)
    symbols = [int(bits[i:i+2], 2) for i in range(0, len(bits), 2)]

    samples_per_symbol = int(samp_rate / baud)
    t = np.arange(samples_per_symbol) / samp_rate

    phase_shifts = [0, np.pi/2, np.pi, 3*np.pi/2]

    out = np.concatenate([np.cos(2 * np.pi * carrier * t + phase_shifts[sym]) for sym in symbols])

    out /= np.max(np.abs(out)) * 0.8
    return out.astype(np.float32)


def qpsk_demodulate(samples: np.ndarray, baud=1200, carrier=3000.0, samp_rate=96000) -> bytes:
    samples_per_symbol = int(samp_rate / baud)

    bits = ''
    for i in range(0, len(samples), samples_per_symbol):
        chunk = samples[i:i+samples_per_symbol]
        phase = np.angle(np.sum(chunk * np.exp(-1j * 2 * np.pi * carrier * np.arange(len(chunk)) / samp_rate)))
        sym = round((phase % (2*np.pi)) / (np.pi/2)) % 4
        bits += f'{sym:02b}'

    bytes_out = [int(bits[j:j+8], 2) for j in range(0, len(bits), 8)]
    return bytes(bytes_out)


def psk8_modulate(data_bytes: bytes, baud=1200, carrier=3000.0, samp_rate=96000) -> np.ndarray:
    bits = ''.join(f'{byte:08b}' for byte in data_bytes)
    symbols = [int(bits[i:i+3], 2) for i in range(0, len(bits), 3)]

    samples_per_symbol = int(samp_rate / baud)
    t = np.arange(samples_per_symbol) / samp_rate

    phase_shifts = [k * np.pi / 4 for k in range(8)]

    out = np.concatenate([np.cos(2 * np.pi * carrier * t + phase_shifts[sym]) for sym in symbols])

    out /= np.max(np.abs(out)) * 0.8
    return out.astype(np.float32)


def psk8_demodulate(samples: np.ndarray, baud=1200, carrier=3000.0, samp_rate=96000) -> bytes:
    samples_per_symbol = int(samp_rate / baud)

    bits = ''
    for i in range(0, len(samples), samples_per_symbol):
        chunk = samples[i:i+samples_per_symbol]
        phase = np.angle(np.sum(chunk * np.exp(-1j * 2 * np.pi * carrier * np.arange(len(chunk)) / samp_rate)))
        sym = round((phase % (2*np.pi)) / (np.pi/4)) % 8
        bits += f'{sym:03b}'

    bits = bits[: (len(bits) // 8) * 8]

    bytes_out = [int(bits[j:j+8], 2) for j in range(0, len(bits), 8)]
    return bytes(bytes_out)


def apsk16_modulate(data_bytes: bytes, baud=1200, carrier=3000.0, samp_rate=96000) -> np.ndarray:
    bits = ''.join(f'{byte:08b}' for byte in data_bytes)
    symbols = [int(bits[i:i+4], 2) for i in range(0, len(bits), 4)]

    samples_per_symbol = int(samp_rate / baud)
    t = np.arange(samples_per_symbol) / samp_rate

    const = [ (a + 1j*b) for a in [-3,-1,1,3] for b in [-3,-1,1,3] ]

    out = np.concatenate([np.real(const[sym] * np.exp(1j * 2 * np.pi * carrier * t)) for sym in symbols])

    out /= np.max(np.abs(out)) * 0.8
    return out.astype(np.float32)


def apsk16_demodulate(samples: np.ndarray, baud=1200, carrier=3000.0, samp_rate=96000) -> bytes:
    samples_per_symbol = int(samp_rate / baud)

    bits = ''
    const = [ (a + 1j*b) for a in [-3,-1,1,3] for b in [-3,-1,1,3] ]

    for i in range(0, len(samples), samples_per_symbol):
        chunk = samples[i:i+samples_per_symbol]
        sym_complex = np.mean(chunk * np.exp(-1j * 2 * np.pi * carrier * np.arange(len(chunk)) / samp_rate))
        sym = np.argmin([np.abs(sym_complex - c) for c in const])
        bits += f'{sym:04b}'

    bytes_out = [int(bits[j:j+8], 2) for j in range(0, len(bits), 8)]
    return bytes(bytes_out)


def dsss_modulate(data_bytes: bytes, baud=1200, carrier=3000.0, samp_rate=96000) -> np.ndarray:
    pn = [1,0,1,1,0,0,1] * (baud // 7)
    bits = [2*int(b)-1 for byte in data_bytes for b in f'{byte:08b}']

    spread = np.repeat(bits, len(pn)) * np.tile(pn, len(bits))

    t = np.arange(len(spread)) / samp_rate
    out = np.sin(2 * np.pi * carrier * t) * (2 * spread - 1)

    out /= np.max(np.abs(out)) * 0.8
    return out.astype(np.float32)


def dsss_demodulate(samples: np.ndarray, baud=1200, carrier=3000.0, samp_rate=96000) -> bytes:
    pn = [1,0,1,1,0,0,1] * (baud // 7)
    chip_rate = len(pn) * baud

    t = np.arange(len(samples)) / samp_rate
    carrier_sig = np.sin(2 * np.pi * carrier * t)

    demod = samples * carrier_sig

    bits = []
    chips_per_bit = len(pn)
    for i in range(0, len(demod), chips_per_bit):
        chunk = demod[i:i+chips_per_bit]
        corr = np.sum(chunk * pn)
        bits.append(1 if corr > 0 else 0)

    bytes_out = [int(''.join(map(str, bits[k:k+8])), 2) for k in range(0, len(bits), 8)]
    return bytes(bytes_out)


def msk_modulate(data_bytes: bytes, baud=1200, carrier=3000.0, samp_rate=96000) -> np.ndarray:
    bits = [2*int(b)-1 for byte in data_bytes for b in f'{byte:08b}']

    samples_per_bit = int(samp_rate / baud)
    t = np.arange(samples_per_bit) / samp_rate

    phase = 0
    out = []
    for bit in bits:
        phase_inc = bit * np.pi / (2 * samples_per_bit)
        phase_arr = phase + np.cumsum(np.ones(samples_per_bit) * phase_inc)
        out.append(np.sin(2 * np.pi * carrier * t + phase_arr))
        phase = phase_arr[-1] % (2*np.pi)

    out = np.concatenate(out)
    out /= np.max(np.abs(out)) * 0.8
    return out.astype(np.float32)


def msk_demodulate(samples: np.ndarray, baud=1200, carrier=3000.0, samp_rate=96000) -> bytes:
    samples_per_bit = int(samp_rate / baud)

    i = samples * np.cos(2 * np.pi * carrier * np.arange(len(samples)) / samp_rate)
    q = samples * np.sin(2 * np.pi * carrier * np.arange(len(samples)) / samp_rate)

    bits = []
    prev_i, prev_q = 0, 0
    for j in range(0, len(samples), samples_per_bit):
        curr_i = np.mean(i[j:j+samples_per_bit])
        curr_q = np.mean(q[j:j+samples_per_bit])
        diff = curr_i * prev_q - curr_q * prev_i
        bits.append(1 if diff > 0 else 0)
        prev_i, prev_q = curr_i, curr_q

    bytes_out = [int(''.join(map(str, bits[k:k+8])), 2) for k in range(0, len(bits), 8)]
    return bytes(bytes_out)


def fsk_high_speed_modulate(data_bytes: bytes, baud=19200, mark_freq=12000.0, space_freq=18000.0,
                              samp_rate=SAMPLE_RATE) -> np.ndarray:
    return fsk_modulate(data_bytes, baud=baud, mark_freq=mark_freq, space_freq=space_freq, samp_rate=samp_rate)


def fsk_high_speed_demodulate(samples: np.ndarray, baud=19200, mark_freq=12000.0, space_freq=18000.0,
                              samp_rate=SAMPLE_RATE) -> bytes:
    lowcut = 8000
    highcut = 22000
    samples = bandpass_filter(samples, lowcut, highcut, samp_rate)

    samples = np.asarray(samples, dtype=np.float32)
    samples_per_bit = int(round(samp_rate / baud))

    def goertzel(chunk, freq):
        s_prev = 0.0
        s_prev2 = 0.0
        normalized = freq / samp_rate
        coeff = 2.0 * math.cos(2 * math.pi * normalized)
        for x in chunk:
            s = x + coeff * s_prev - s_prev2
            s_prev2 = s_prev
            s_prev = s
        power = s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2
        return power

    bits = []
    for i in range(0, len(samples) - samples_per_bit + 1, samples_per_bit):
        chunk = samples[i:i + samples_per_bit]
        p_mark = goertzel(chunk, mark_freq)
        p_space = goertzel(chunk, space_freq)
        bits.append(1 if p_mark > p_space else 0)

    out = []
    for i in range(0, len(bits) - 7, 8):
        byte_bits = bits[i:i + 8]
        b = 0
        for k, bit in enumerate(byte_bits):
            b |= (bit & 1) << (7 - k)
        out.append(b)

    return bytes(out)


def ofdm_modulate_simple(data_bytes: bytes, baud=4800, carrier=12000.0, num_subcarriers=8,
                         samp_rate=SAMPLE_RATE) -> np.ndarray:
    from scipy.fft import ifft

    bits = ''.join(format(b, '08b') for b in data_bytes)

    bits_per_symbol = num_subcarriers * 2
    symbols = []

    for i in range(0, len(bits), bits_per_symbol):
        symbol_bits = bits[i:i + bits_per_symbol]
        if len(symbol_bits) < bits_per_symbol:
            symbol_bits += '0' * (bits_per_symbol - len(symbol_bits))

        symbol = []
        for j in range(0, len(symbol_bits), 2):
            two_bits = symbol_bits[j:j + 2]
            if two_bits == '00':
                symbol.append(1 + 1j)
            elif two_bits == '01':
                symbol.append(-1 + 1j)
            elif two_bits == '10':
                symbol.append(1 - 1j)
            else:
                symbol.append(-1 - 1j)

        symbols.append(symbol)

    time_domain = []
    samples_per_symbol = int(round(samp_rate / baud))

    for symbol in symbols:
        ifft_input = np.zeros(num_subcarriers, dtype=complex)
        ifft_input[1:len(symbol) + 1] = symbol

        td_symbol = ifft(ifft_input)

        cp_length = samples_per_symbol // 4
        cyclic_prefix = td_symbol[-cp_length:]
        full_symbol = np.concatenate([cyclic_prefix, td_symbol])

        time_domain.extend(full_symbol.real)

    output = np.array(time_domain, dtype=np.float32)
    m = np.max(np.abs(output))
    if m > 0:
        output = output / m * 0.8
    return output


def ofdm_demodulate_simple(samples: np.ndarray, baud=4800, carrier=12000.0, num_subcarriers=8,
                           samp_rate=SAMPLE_RATE) -> bytes:
    from scipy.fft import fft

    samples_per_symbol = int(round(samp_rate / baud))
    cp_length = samples_per_symbol // 4
    useful_length = samples_per_symbol - cp_length

    bits = ''
    i = 0

    while i + samples_per_symbol <= len(samples):
        # Remove cyclic prefix
        symbol_with_cp = samples[i:i + samples_per_symbol]
        symbol = symbol_with_cp[cp_length:cp_length + useful_length]

        # FFT
        freq_domain = fft(symbol)

        # Extrai subportadoras (ignora DC e extremos)
        subcarriers = freq_domain[1:num_subcarriers + 1]

        # Demodula QPSK em cada subportadora
        for sc in subcarriers:
            I, Q = sc.real, sc.imag

            if I >= 0 and Q >= 0:
                bits += '00'
            elif I < 0 and Q >= 0:
                bits += '01'
            elif I < 0 and Q < 0:
                bits += '11'
            else:
                bits += '10'

        i += samples_per_symbol

    if len(bits) % 8 != 0:
        bits = bits[:-(len(bits) % 8)]

    out = bytearray()
    for i in range(0, len(bits), 8):
        out.append(int(bits[i:i + 8], 2))
    return bytes(out)


def ft8_modulate(data_bytes: bytes, baud=6.25, carrier=1500.0, samp_rate=SAMPLE_RATE) -> np.ndarray:
    return ofdm_modulate_simple(data_bytes, baud=baud, carrier=carrier, num_subcarriers=79)


def ft8_demodulate(samples: np.ndarray, baud=6.25, carrier=1500.0, samp_rate=SAMPLE_RATE) -> bytes:
    return ofdm_demodulate_simple(samples, baud=baud, carrier=carrier, num_subcarriers=79)


def ft4_modulate(data_bytes: bytes, baud=20.83, carrier=1500.0, samp_rate=SAMPLE_RATE) -> np.ndarray:
    return ft8_modulate(data_bytes, baud=baud, carrier=carrier)


def ft4_demodulate(samples: np.ndarray, baud=20.83, carrier=1500.0, samp_rate=SAMPLE_RATE) -> bytes:
    return ft8_demodulate(samples, baud=baud, carrier=carrier)


def js8_modulate(data_bytes: bytes, baud=15.625, carrier=1500.0, samp_rate=SAMPLE_RATE) -> np.ndarray:
    return ft8_modulate(data_bytes, baud=baud, carrier=carrier)


def js8_demodulate(samples: np.ndarray, baud=15.625, carrier=1500.0, samp_rate=SAMPLE_RATE) -> bytes:
    return ft8_demodulate(samples, baud=baud, carrier=carrier)


def psk31_modulate(data_bytes: bytes, baud=31.25, carrier=1000.0, samp_rate=SAMPLE_RATE) -> np.ndarray:
    return bpsk_modulate(data_bytes, baud=baud, carrier=carrier)


def psk31_demodulate(samples: np.ndarray, baud=31.25, carrier=1000.0, samp_rate=SAMPLE_RATE) -> bytes:
    return bpsk_demodulate(samples, baud=baud, carrier=carrier)


def rtty_modulate(data_bytes: bytes, baud=45, mark_freq=2295.0, space_freq=2125.0, samp_rate=SAMPLE_RATE) -> np.ndarray:
    return fsk_modulate(data_bytes, baud=baud, mark_freq=mark_freq, space_freq=space_freq, samp_rate=samp_rate)


def rtty_demodulate(samples: np.ndarray, baud=45, mark_freq=2295.0, space_freq=2125.0, samp_rate=SAMPLE_RATE) -> bytes:
    return fsk_demodulate(samples, baud=baud, mark_freq=mark_freq, space_freq=space_freq, samp_rate=samp_rate)


def mfsk8_modulate(data_bytes: bytes, baud=7.8125, carrier=1000.0, num_tones=8, samp_rate=SAMPLE_RATE) -> np.ndarray:
    return ofdm_modulate_simple(data_bytes, baud=baud, carrier=carrier, num_subcarriers=num_tones)


def mfsk8_demodulate(samples: np.ndarray, baud=7.8125, carrier=1000.0, num_tones=8, samp_rate=SAMPLE_RATE) -> bytes:
    return ofdm_demodulate_simple(samples, baud=baud, carrier=carrier, num_subcarriers=num_tones)


def afsk1200_modulate(data_bytes: bytes, baud=1200, mark_freq=1200.0, space_freq=2200.0, samp_rate=SAMPLE_RATE) -> np.ndarray:
    return fsk_modulate(data_bytes, baud=baud, mark_freq=mark_freq, space_freq=space_freq, samp_rate=samp_rate)


def afsk1200_demodulate(samples: np.ndarray, baud=1200, mark_freq=1200.0, space_freq=2200.0, samp_rate=SAMPLE_RATE) -> bytes:
    return fsk_demodulate(samples, baud=baud, mark_freq=mark_freq, space_freq=space_freq, samp_rate=samp_rate)


def pactor_modulate(data_bytes: bytes, baud=200, carrier=1500.0, samp_rate=SAMPLE_RATE) -> np.ndarray:
    return psk8_modulate(data_bytes, baud=baud, carrier=carrier)


def pactor_demodulate(samples: np.ndarray, baud=200, carrier=1500.0, samp_rate=SAMPLE_RATE) -> bytes:
    return psk8_demodulate(samples, baud=baud, carrier=carrier)


def dmr_modulate(data_bytes: bytes, baud=4800, carrier=3000.0, samp_rate=SAMPLE_RATE) -> np.ndarray:
    return apsk16_modulate(data_bytes, baud=baud, carrier=carrier)


def dmr_demodulate(samples: np.ndarray, baud=4800, carrier=3000.0, samp_rate=SAMPLE_RATE) -> bytes:
    return apsk16_demodulate(samples, baud=baud, carrier=carrier)


def olivia_modulate(data_bytes: bytes, baud=31.25, carrier=1000.0, num_tones=32, samp_rate=SAMPLE_RATE) -> np.ndarray:
    return mfsk8_modulate(data_bytes, baud=baud, carrier=carrier, num_tones=num_tones)


def olivia_demodulate(samples: np.ndarray, baud=31.25, carrier=1000.0, num_tones=32, samp_rate=SAMPLE_RATE) -> bytes:
    return mfsk8_demodulate(samples, baud=baud, carrier=carrier, num_tones=num_tones)


def feld_hell_modulate(data_bytes: bytes, baud=122.5, carrier=1000.0, samp_rate=SAMPLE_RATE) -> np.ndarray:
    return hellschreiber_modulate(data_bytes.decode('utf-8'), baud=baud, carrier=carrier, samp_rate=samp_rate)


def feld_hell_demodulate(samples: np.ndarray, baud=122.5, carrier=1000.0, samp_rate=SAMPLE_RATE) -> bytes:
    return hellschreiber_demodulate(samples, baud=baud, carrier=carrier, samp_rate=samp_rate).encode('utf-8')


def lora_modulate(data_bytes: bytes, spreading_factor=7, bandwidth=125000, samp_rate=SAMPLE_RATE) -> np.ndarray:
    t = np.arange(len(data_bytes) * 100) / samp_rate
    chirp = np.sin(2 * np.pi * (bandwidth / 2) * t ** 2)
    return chirp.astype(np.float32)


def lora_demodulate(samples: np.ndarray, spreading_factor=7, bandwidth=125000, samp_rate=SAMPLE_RATE) -> bytes:
    return b''


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.lfilter(b, a, data)


# --- helpers to write WAV ---
def wav_from_array(arr: np.ndarray, samp_rate=SAMPLE_RATE) -> bytes:
    a = (arr * 32767).astype(np.int16)
    bio = io.BytesIO()
    wf = wave.open(bio, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(samp_rate)
    wf.writeframes(a.tobytes())
    wf.close()
    bio.seek(0)
    return bio.read()