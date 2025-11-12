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
        """Filtro adaptativo com ajuste automático de parâmetros"""
        if not self.noise_reduction:
            return data

        nyquist = self.sample_rate / 2
        normal_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(6, normal_cutoff, btype='low')
        return signal.filtfilt(b, a, data)

    def _calculate_snr(self, signal_data: np.ndarray) -> float:
        """Calcula SNR em tempo real"""
        if len(signal_data) < 1000:
            return 30.0  # Valor padrão para sinais curtos

        # Estima ruído usando filtro high-pass
        b, a = signal.butter(3, 0.1, btype='high')
        noise_estimate = signal.filtfilt(b, a, signal_data)

        signal_power = np.mean(signal_data ** 2)
        noise_power = np.mean(noise_estimate ** 2)

        return 10 * np.log10(signal_power / (noise_power + 1e-10)) if noise_power > 0 else 50.0

    def _adaptive_gain_control(self, data: np.ndarray) -> np.ndarray:
        """Controle automático de ganho"""
        rms = np.sqrt(np.mean(data ** 2))
        if rms > 0:
            target_rms = 0.3
            gain = target_rms / rms
            return data * min(gain, 5.0)  # Limitar ganho máximo
        return data


# Funções de modulação otimizadas com processamento em lote
def fsk_modulate_optimized(data_bytes: bytes, baud=1200, mark_freq=1200.0,
                           space_freq=2200.0, samp_rate=96000) -> np.ndarray:
    modem = AdvancedModem()

    print(f"🎵 Modulando FSK Otimizada: {len(data_bytes)} bytes, baud={baud}")

    if baud >= 9600:
        mark_freq = 8000.0
        space_freq = 16000.0

    bit_dur = 1.0 / baud
    samples_per_bit = int(round(samp_rate * bit_dur))

    # Pré-computar ondas para melhor performance
    t = np.arange(samples_per_bit) / samp_rate
    mark_wave = np.sin(2 * np.pi * mark_freq * t)
    space_wave = np.sin(2 * np.pi * space_freq * t)
    sync_wave = np.sin(2 * np.pi * space_freq * t)  # Onda de sincronismo

    # Converter dados para bits
    bits = []
    for byte in data_bytes:
        bits.extend([int(b) for b in f'{byte:08b}'])

    # Criar array de saída pré-alocado
    total_samples = samples_per_bit * (len(bits) + 2)  # +2 para sync bits
    out = np.zeros(total_samples, dtype=np.float32)

    # Adicionar bits de sincronismo
    out[:samples_per_bit] = sync_wave
    ptr = samples_per_bit

    # Modular bits
    for bit in bits:
        out[ptr:ptr + samples_per_bit] = mark_wave if bit else space_wave
        ptr += samples_per_bit

    # Finalizar com sync
    out[ptr:ptr + samples_per_bit] = sync_wave

    # Aplicar processamento adaptativo
    out = modem._adaptive_gain_control(out)

    # Normalizar
    m = np.max(np.abs(out))
    if m > 0:
        out = out / m * 0.8

    return out


# Demodulação melhorada com detecção adaptativa
def fsk_demodulate_enhanced(samples: np.ndarray, baud=1200, mark_freq=1200.0,
                            space_freq=2200.0, samp_rate=96000) -> bytes:
    modem = AdvancedModem()

    print(f"🔍 Demodulando FSK Avançada: {len(samples)} amostras")

    if baud >= 9600:
        mark_freq = 8000.0
        space_freq = 16000.0

    # Aplicar filtragem adaptativa baseada no SNR
    snr = modem._calculate_snr(samples)
    cutoff = min(space_freq + 2000, samp_rate / 2 * 0.9)

    if snr < 20:  # SNR baixo, filtrar mais agressivamente
        cutoff = space_freq + 1000

    filtered = modem._adaptive_filter(samples, cutoff)

    samples_per_bit = int(round(samp_rate / baud))

    # Detector Goertzel otimizado
    def optimized_goertzel(chunk, target_freq):
        w = 2.0 * math.pi * target_freq / samp_rate
        cos_w = math.cos(w)
        coeff = 2.0 * cos_w

        s_prev, s_prev2 = 0.0, 0.0
        for x in chunk:
            s = x + coeff * s_prev - s_prev2
            s_prev2, s_prev = s_prev, s

        return s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2

    # Processamento em lote para melhor performance
    bits = []
    for i in range(0, len(filtered) - samples_per_bit + 1, samples_per_bit):
        chunk = filtered[i:i + samples_per_bit]
        p_mark = optimized_goertzel(chunk, mark_freq)
        p_space = optimized_goertzel(chunk, space_freq)
        bits.append(1 if p_mark > p_space else 0)

    # Decodificação robusta com correção de sincronismo
    out = []
    i = 0
    sync_found = False

    while i + 10 <= len(bits):
        # Buscar sincronismo
        if not sync_found and bits[i] == 0:
            sync_found = True

        if sync_found:
            byte_bits = bits[i:i + 8]
            b = sum(bit << j for j, bit in enumerate(byte_bits))
            out.append(b)
            i += 8
        else:
            i += 1

    print(f"📨 Bytes reconstruídos: {len(out)}")
    return bytes(out)


# Adicionar modulação QAM16 para maior eficiência espectral
def qam16_modulate(data_bytes: bytes, baud=2400, carrier=6000.0, samp_rate=96000) -> np.ndarray:
    """Modulação QAM16 para alta eficiência espectral"""
    modem = AdvancedModem()

    # Mapeamento QAM16
    symbol_map = {
        (0, 0, 0, 0): (-3, -3), (0, 0, 0, 1): (-3, -1), (0, 0, 1, 0): (-3, 3), (0, 0, 1, 1): (-3, 1),
        (0, 1, 0, 0): (-1, -3), (0, 1, 0, 1): (-1, -1), (0, 1, 1, 0): (-1, 3), (0, 1, 1, 1): (-1, 1),
        (1, 0, 0, 0): (3, -3), (1, 0, 0, 1): (3, -1), (1, 0, 1, 0): (3, 3), (1, 0, 1, 1): (3, 1),
        (1, 1, 0, 0): (1, -3), (1, 1, 0, 1): (1, -1), (1, 1, 1, 0): (1, 3), (1, 1, 1, 1): (1, 1)
    }

    bits = ''.join(format(b, '08b') for b in data_bytes)
    if len(bits) % 4 != 0:
        bits += '0' * (4 - len(bits) % 4)

    symbols = [bits[i:i + 4] for i in range(0, len(bits), 4)]
    samples_per_symbol = int(round(samp_rate / baud))
    t = np.arange(samples_per_symbol) / samp_rate

    out = np.zeros(samples_per_symbol * len(symbols), dtype=np.float32)

    for i, symbol_bits in enumerate(symbols):
        I, Q = symbol_map[tuple(int(b) for b in symbol_bits)]
        carrier_i = np.cos(2 * math.pi * carrier * t)
        carrier_q = np.sin(2 * math.pi * carrier * t)
        out[i * samples_per_symbol:(i + 1) * samples_per_symbol] = I * carrier_i + Q * carrier_q

    out = modem._adaptive_gain_control(out)
    m = np.max(np.abs(out))
    if m > 0:
        out = out / m * 0.8

    return out


# Atualizar mapa de modulação com novos métodos
MODULATION_MAP = {
    "FSK1200": (fsk_modulate_optimized, fsk_demodulate_enhanced),
    "FSK9600": (fsk_modulate_optimized, fsk_demodulate_enhanced),
    "QAM16": (qam16_modulate, None),  # Demodulação QAM seria implementada
    # ... outras modulações
}

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def fsk_modulate(data_bytes: bytes, baud=1200, mark_freq=1200.0, space_freq=2200.0,
                 samp_rate=SAMPLE_RATE, quality_indicator: float = 1.0) -> np.ndarray:
    print(f"🎵 Modulando FSK: {len(data_bytes)} bytes, baud={baud}, mark={mark_freq}, space={space_freq}")

    if baud >= 9600:
        mark_freq = 8000.0
        space_freq = 16000.0

    bit_dur = 1.0 / baud
    samples_per_bit = int(round(samp_rate * bit_dur))
    total_bits = len(data_bytes) * 10
    t = np.arange(samples_per_bit) / samp_rate
    out = np.zeros(samples_per_bit * total_bits, dtype=np.float32)
    ptr = 0

    for b in data_bytes:
        out[ptr:ptr + samples_per_bit] = np.sin(2 * np.pi * space_freq * t)
        ptr += samples_per_bit

        for i in range(8):
            bit = (b >> i) & 1
            f = mark_freq if bit else space_freq
            out[ptr:ptr + samples_per_bit] = np.sin(2 * np.pi * f * t)
            ptr += samples_per_bit

        out[ptr:ptr + samples_per_bit] = np.sin(2 * np.pi * mark_freq * t)
        ptr += samples_per_bit

    m = np.max(np.abs(out))
    if m > 0:
        out = out / m * 0.8
    return out


def fsk_demodulate(samples: np.ndarray, baud=1200, mark_freq=1200.0, space_freq=2200.0, samp_rate=SAMPLE_RATE) -> bytes:
    print(f"🔍 Demodulando FSK: {len(samples)} amostras, baud={baud}")

    if baud >= 9600:
        mark_freq = 8000.0
        space_freq = 16000.0

    lowcut = min(mark_freq, space_freq) - 500
    highcut = max(mark_freq, space_freq) + 500
    samples = bandpass_filter(samples, lowcut, highcut, samp_rate)

    samples = np.asarray(samples, dtype=np.float32)
    samples_per_bit = int(round(samp_rate / baud))
    print(f"📊 Samples per bit: {samples_per_bit}")

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
    bit_count = 0
    for i in range(0, len(samples) - samples_per_bit + 1, samples_per_bit):
        chunk = samples[i:i + samples_per_bit]
        p_mark = goertzel(chunk, mark_freq)
        p_space = goertzel(chunk, space_freq)
        bits.append(1 if p_mark > p_space else 0)
        bit_count += 1

    print(f"🔢 Total de bits detectados: {bit_count}")

    out = []
    i = 0
    while i + 10 <= len(bits):
        if bits[i] != 0:
            i += 1
            continue
        b = 0
        for k in range(8):
            b |= (bits[i + 1 + k] & 1) << k
        stop = bits[i + 9]
        if stop != 1:
            i += 1
            continue
        out.append(b)
        i += 10

    print(f"📨 Bytes reconstruídos: {len(out)}")
    return bytes(out)


def bpsk_modulate(data_bytes: bytes, baud=1200, carrier=3000.0, samp_rate=SAMPLE_RATE) -> np.ndarray:
    bits = ''.join(format(b, '08b') for b in data_bytes)
    samples_per_bit = int(round(samp_rate / baud))
    t = np.arange(samples_per_bit) / samp_rate
    out = np.zeros(samples_per_bit * len(bits), dtype=np.float32)
    ptr = 0

    for bit in bits:
        phase = 0.0 if bit == '0' else math.pi
        out[ptr:ptr + samples_per_bit] = np.sin(2 * math.pi * carrier * t + phase)
        ptr += samples_per_bit

    m = np.max(np.abs(out))
    if m > 0:
        out = out / m * 0.8
    return out


def bpsk_demodulate(samples: np.ndarray, baud=1200, carrier=3000.0, samp_rate=SAMPLE_RATE) -> bytes:
    samples = np.asarray(samples, dtype=np.float32)
    samples_per_bit = int(round(samp_rate / baud))
    bits = []
    t = np.arange(samples_per_bit) / samp_rate
    ref_cos = np.cos(2 * math.pi * carrier * t)
    ref_sin = np.sin(2 * math.pi * carrier * t)

    for i in range(0, len(samples) - samples_per_bit + 1, samples_per_bit):
        chunk = samples[i:i + samples_per_bit]
        i_val = np.sum(chunk * ref_cos)
        q_val = np.sum(chunk * ref_sin)
        bits.append('0' if i_val > 0 else '1')

    bitstr = ''.join(bits)
    if len(bitstr) % 8 != 0:
        bitstr = bitstr[:-(len(bitstr) % 8)]

    out = bytearray()
    for i in range(0, len(bitstr), 8):
        out.append(int(bitstr[i:i + 8], 2))
    return bytes(out)


def qpsk_modulate(data_bytes: bytes, baud=1200, carrier=3000.0, samp_rate=SAMPLE_RATE) -> np.ndarray:
    bits = ''.join(format(b, '08b') for b in data_bytes)
    if len(bits) % 2 != 0:
        bits += '0'

    symbols = [bits[i:i + 2] for i in range(0, len(bits), 2)]
    samples_per_symbol = int(round(samp_rate / baud))
    t = np.arange(samples_per_symbol) / samp_rate
    out = np.zeros(samples_per_symbol * len(symbols), dtype=np.float32)
    ptr = 0

    map_phase = {'00': 0.0, '01': math.pi / 2, '11': math.pi, '10': 3 * math.pi / 2}

    for sym in symbols:
        phase = map_phase.get(sym, 0.0)
        out[ptr:ptr + samples_per_symbol] = np.sin(2 * math.pi * carrier * t + phase)
        ptr += samples_per_symbol

    m = np.max(np.abs(out))
    if m > 0:
        out = out / m * 0.8
    return out


def qpsk_demodulate(samples: np.ndarray, baud=1200, carrier=3000.0, samp_rate=SAMPLE_RATE) -> bytes:
    samples = np.asarray(samples, dtype=np.float32)
    samples_per_symbol = int(round(samp_rate / baud))
    t = np.arange(samples_per_symbol) / samp_rate
    cos_ref = np.cos(2 * math.pi * carrier * t)
    sin_ref = np.sin(2 * math.pi * carrier * t)
    bits = ''

    for i in range(0, len(samples) - samples_per_symbol + 1, samples_per_symbol):
        chunk = samples[i:i + samples_per_symbol]
        I = np.sum(chunk * cos_ref)
        Q = np.sum(chunk * sin_ref)

        if I >= 0 and Q >= 0:
            bits += '00'
        elif I < 0 and Q >= 0:
            bits += '01'
        elif I < 0 and Q < 0:
            bits += '11'
        else:
            bits += '10'

    if len(bits) % 8 != 0:
        bits = bits[:-(len(bits) % 8)]

    out = bytearray()
    for i in range(0, len(bits), 8):
        out.append(int(bits[i:i + 8], 2))
    return bytes(out)


def psk8_modulate(data_bytes: bytes, baud=2400, carrier=12000.0, samp_rate=SAMPLE_RATE) -> np.ndarray:
    bits = ''.join(format(b, '08b') for b in data_bytes)
    if len(bits) % 3 != 0:
        bits += '0' * (3 - len(bits) % 3)

    symbols = [bits[i:i + 3] for i in range(0, len(bits), 3)]
    samples_per_symbol = int(round(samp_rate / baud))
    t = np.arange(samples_per_symbol) / samp_rate
    out = np.zeros(samples_per_symbol * len(symbols), dtype=np.float32)
    ptr = 0

    map_phase = {
        '000': 0.0, '001': math.pi / 4, '010': math.pi / 2, '011': 3 * math.pi / 4,
        '100': math.pi, '101': 5 * math.pi / 4, '110': 3 * math.pi / 2, '111': 7 * math.pi / 4
    }

    for sym in symbols:
        phase = map_phase.get(sym, 0.0)
        out[ptr:ptr + samples_per_symbol] = np.sin(2 * math.pi * carrier * t + phase)
        ptr += samples_per_symbol

    m = np.max(np.abs(out))
    if m > 0:
        out = out / m * 0.8
    return out


def psk8_demodulate(samples: np.ndarray, baud=2400, carrier=12000.0, samp_rate=SAMPLE_RATE) -> bytes:
    samples = np.asarray(samples, dtype=np.float32)
    samples_per_symbol = int(round(samp_rate / baud))
    t = np.arange(samples_per_symbol) / samp_rate
    cos_ref = np.cos(2 * math.pi * carrier * t)
    sin_ref = np.sin(2 * math.pi * carrier * t)
    bits = ''

    for i in range(0, len(samples) - samples_per_symbol + 1, samples_per_symbol):
        chunk = samples[i:i + samples_per_symbol]
        I = np.sum(chunk * cos_ref)
        Q = np.sum(chunk * sin_ref)

        phase = math.atan2(Q, I)
        if phase < 0:
            phase += 2 * math.pi

        if phase < math.pi / 8:
            bits += '000'
        elif phase < 3 * math.pi / 8:
            bits += '001'
        elif phase < 5 * math.pi / 8:
            bits += '010'
        elif phase < 7 * math.pi / 8:
            bits += '011'
        elif phase < 9 * math.pi / 8:
            bits += '100'
        elif phase < 11 * math.pi / 8:
            bits += '101'
        elif phase < 13 * math.pi / 8:
            bits += '110'
        else:
            bits += '111'

    if len(bits) % 8 != 0:
        bits = bits[:-(len(bits) % 8)]

    out = bytearray()
    for i in range(0, len(bits), 8):
        out.append(int(bits[i:i + 8], 2))
    return bytes(out)


def fsk_high_speed_modulate(data_bytes: bytes, baud=19200, mark_freq=12000.0, space_freq=18000.0,
                            samp_rate=SAMPLE_RATE) -> np.ndarray:
    bit_dur = 1.0 / baud
    samples_per_bit = int(round(samp_rate * bit_dur))
    t = np.arange(samples_per_bit) / samp_rate

    mark_wave = np.sin(2 * np.pi * mark_freq * t)
    space_wave = np.sin(2 * np.pi * space_freq * t)

    bits = ''.join(format(b, '08b') for b in data_bytes)
    out = np.zeros(samples_per_bit * len(bits), dtype=np.float32)

    for i, bit in enumerate(bits):
        start = i * samples_per_bit
        end = start + samples_per_bit
        out[start:end] = mark_wave if bit == '1' else space_wave

    m = np.max(np.abs(out))
    if m > 0:
        out = out / m * 0.8
    return out


def apsk16_modulate(data_bytes: bytes, baud=3200, carrier=12000.0, samp_rate=SAMPLE_RATE) -> np.ndarray:
    """Modulação APSK16 - combinação de amplitude e fase"""
    bits = ''.join(format(b, '08b') for b in data_bytes)
    if len(bits) % 4 != 0:
        bits += '0' * (4 - len(bits) % 4)

    symbols = [bits[i:i + 4] for i in range(0, len(bits), 4)]
    samples_per_symbol = int(round(samp_rate / baud))
    t = np.arange(samples_per_symbol) / samp_rate

    # Constelação APSK16: 2 amplitudes, 8 fases
    constellation = {
        '0000': (0.5, 0), '0001': (0.5, math.pi / 4),
        '0010': (0.5, math.pi / 2), '0011': (0.5, 3 * math.pi / 4),
        '0100': (0.5, math.pi), '0101': (0.5, 5 * math.pi / 4),
        '0110': (0.5, 3 * math.pi / 2), '0111': (0.5, 7 * math.pi / 4),
        '1000': (1.0, 0), '1001': (1.0, math.pi / 4),
        '1010': (1.0, math.pi / 2), '1011': (1.0, 3 * math.pi / 4),
        '1100': (1.0, math.pi), '1101': (1.0, 5 * math.pi / 4),
        '1110': (1.0, 3 * math.pi / 2), '1111': (1.0, 7 * math.pi / 4)
    }

    out = np.zeros(samples_per_symbol * len(symbols), dtype=np.float32)
    ptr = 0

    for sym in symbols:
        amp, phase = constellation.get(sym, (1.0, 0.0))
        out[ptr:ptr + samples_per_symbol] = amp * np.sin(2 * math.pi * carrier * t + phase)
        ptr += samples_per_symbol

    m = np.max(np.abs(out))
    if m > 0:
        out = out / m * 0.8
    return out


def apsk16_demodulate(samples: np.ndarray, baud=3200, carrier=12000.0, samp_rate=SAMPLE_RATE) -> bytes:
    """Demodulação APSK16"""
    samples = np.asarray(samples, dtype=np.float32)
    samples_per_symbol = int(round(samp_rate / baud))
    t = np.arange(samples_per_symbol) / samp_rate

    cos_ref = np.cos(2 * math.pi * carrier * t)
    sin_ref = np.sin(2 * math.pi * carrier * t)

    bits = ''

    for i in range(0, len(samples) - samples_per_symbol + 1, samples_per_symbol):
        chunk = samples[i:i + samples_per_symbol]
        I = np.sum(chunk * cos_ref)
        Q = np.sum(chunk * sin_ref)

        amp = np.sqrt(I ** 2 + Q ** 2)
        phase = math.atan2(Q, I)
        if phase < 0:
            phase += 2 * math.pi

        # Decisão baseada na constelação
        if amp < 0.75:  # Anel interno
            if phase < math.pi / 8:
                bits += '0000'
            elif phase < 3 * math.pi / 8:
                bits += '0001'
            elif phase < 5 * math.pi / 8:
                bits += '0010'
            elif phase < 7 * math.pi / 8:
                bits += '0011'
            elif phase < 9 * math.pi / 8:
                bits += '0100'
            elif phase < 11 * math.pi / 8:
                bits += '0101'
            elif phase < 13 * math.pi / 8:
                bits += '0110'
            else:
                bits += '0111'
        else:  # Anel externo
            if phase < math.pi / 8:
                bits += '1000'
            elif phase < 3 * math.pi / 8:
                bits += '1001'
            elif phase < 5 * math.pi / 8:
                bits += '1010'
            elif phase < 7 * math.pi / 8:
                bits += '1011'
            elif phase < 9 * math.pi / 8:
                bits += '1100'
            elif phase < 11 * math.pi / 8:
                bits += '1101'
            elif phase < 13 * math.pi / 8:
                bits += '1110'
            else:
                bits += '1111'

    if len(bits) % 8 != 0:
        bits = bits[:-(len(bits) % 8)]

    out = bytearray()
    for i in range(0, len(bits), 8):
        out.append(int(bits[i:i + 8], 2))
    return bytes(out)


def dsss_modulate(data_bytes: bytes, baud=1200, chip_rate=9600, carrier=3000.0,
                  samp_rate=SAMPLE_RATE, spreading_code=None) -> np.ndarray:
    """Modulação DSSS com espalhamento espectral"""
    if spreading_code is None:
        # Código de espalhamento padrão (sequência pseudo-aleatória)
        spreading_code = [1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0]

    chips_per_bit = len(spreading_code)
    samples_per_chip = int(round(samp_rate / chip_rate))

    # Converter dados para bits
    bits = []
    for byte in data_bytes:
        bits.extend([int(b) for b in f'{byte:08b}'])

    # Espalhar cada bit com o código
    chips = []
    for bit in bits:
        for chip in spreading_code:
            chips.append(bit ^ chip)  # XOR para espalhamento

    # Modular com BPSK
    t = np.arange(samples_per_chip) / samp_rate
    out = np.zeros(samples_per_chip * len(chips), dtype=np.float32)
    ptr = 0

    for chip in chips:
        phase = 0.0 if chip == 1 else math.pi
        out[ptr:ptr + samples_per_chip] = np.sin(2 * math.pi * carrier * t + phase)
        ptr += samples_per_chip

    m = np.max(np.abs(out))
    if m > 0:
        out = out / m * 0.8
    return out


def dsss_demodulate(samples: np.ndarray, baud=1200, chip_rate=9600, carrier=3000.0,
                    samp_rate=SAMPLE_RATE, spreading_code=None) -> bytes:
    """Demodulação DSSS"""
    if spreading_code is None:
        spreading_code = [1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0]

    chips_per_bit = len(spreading_code)
    samples_per_chip = int(round(samp_rate / chip_rate))

    # Demodular BPSK primeiro
    samples = np.asarray(samples, dtype=np.float32)
    t = np.arange(samples_per_chip) / samp_rate
    ref_cos = np.cos(2 * math.pi * carrier * t)

    chips = []
    for i in range(0, len(samples) - samples_per_chip + 1, samples_per_chip):
        chunk = samples[i:i + samples_per_chip]
        correlation = np.sum(chunk * ref_cos)
        chips.append(1 if correlation > 0 else 0)

    # Desspreading
    bits = []
    for i in range(0, len(chips) - chips_per_bit + 1, chips_per_bit):
        chip_group = chips[i:i + chips_per_bit]

        # Correlacionar com código de espalhamento
        correlations = []
        for chip_bit, spread_bit in zip(chip_group, spreading_code):
            correlations.append(1 if chip_bit == spread_bit else -1)

        bit_decision = sum(correlations)
        bits.append('1' if bit_decision > 0 else '0')

    bitstr = ''.join(bits)
    if len(bitstr) % 8 != 0:
        bitstr = bitstr[:-(len(bitstr) % 8)]

    out = bytearray()
    for i in range(0, len(bitstr), 8):
        out.append(int(bitstr[i:i + 8], 2))
    return bytes(out)


def msk_modulate(data_bytes: bytes, baud=2400, carrier=6000.0, samp_rate=SAMPLE_RATE) -> np.ndarray:
    """Modulação MSK - FSK com índice de modulação 0.5"""
    bits = ''.join(format(b, '08b') for b in data_bytes)

    samples_per_bit = int(round(samp_rate / baud))
    total_samples = samples_per_bit * len(bits)
    out = np.zeros(total_samples, dtype=np.float32)

    # Frequências para 0 e 1 (índice de modulação 0.5)
    f0 = carrier - baud / 4
    f1 = carrier + baud / 4

    phase = 0.0
    for i, bit in enumerate(bits):
        freq = f1 if bit == '1' else f0
        start_sample = i * samples_per_bit
        end_sample = start_sample + samples_per_bit

        t = np.arange(samples_per_bit) / samp_rate
        phase_cont = phase + 2 * math.pi * freq * t

        out[start_sample:end_sample] = np.sin(phase_cont)

        # Atualizar fase continuamente
        phase += 2 * math.pi * freq * samples_per_bit / samp_rate
        phase %= 2 * math.pi

    m = np.max(np.abs(out))
    if m > 0:
        out = out / m * 0.8
    return out


def msk_demodulate(samples: np.ndarray, baud=2400, carrier=6000.0, samp_rate=SAMPLE_RATE) -> bytes:
    """Demodulação MSK"""
    samples = np.asarray(samples, dtype=np.float32)
    samples_per_bit = int(round(samp_rate / baud))

    f0 = carrier - baud / 4
    f1 = carrier + baud / 4

    bits = []
    for i in range(0, len(samples) - samples_per_bit + 1, samples_per_bit):
        chunk = samples[i:i + samples_per_bit]

        # Detectar frequência usando Goertzel
        def goertzel_freq_detect(data, target_freq):
            w = 2.0 * math.pi * target_freq / samp_rate
            coeff = 2.0 * math.cos(w)
            s_prev, s_prev2 = 0.0, 0.0
            for x in data:
                s = x + coeff * s_prev - s_prev2
                s_prev2, s_prev = s_prev, s
            return s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2

        p0 = goertzel_freq_detect(chunk, f0)
        p1 = goertzel_freq_detect(chunk, f1)

        bits.append('1' if p1 > p0 else '0')

    bitstr = ''.join(bits)
    if len(bitstr) % 8 != 0:
        bitstr = bitstr[:-(len(bitstr) % 8)]

    out = bytearray()
    for i in range(0, len(bitstr), 8):
        out.append(int(bitstr[i:i + 8], 2))
    return bytes(out)

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