# decoder.py - VERSÃO MELHORADA COM CONTROLE DE DUPLICATAS E ORDEM
import os, time, struct, binascii, threading
from collections import defaultdict, deque
from datetime import datetime
import sounddevice as sd
import numpy as np
import soundfile as sf
from sklearn.ensemble import IsolationForest
import joblib

from modem import (fsk_demodulate, bpsk_demodulate, qpsk_demodulate,
                   psk8_demodulate, fsk_high_speed_demodulate, ofdm_demodulate_simple,
                   SAMPLE_RATE, ft8_demodulate, psk31_demodulate, feld_hell_demodulate)
from utils.compression import decompress_data, super_decompress, delta_decompress


class FileAssembly:
    def __init__(self, filename: str, total_parts: int, file_size: int, file_crc: int):
        self.filename = filename
        self.total_parts = total_parts
        self.file_size = file_size
        self.expected_crc = file_crc
        self.parts = [None] * total_parts
        self.parts_quality = [0.0] * total_parts
        self.received_parts = 0
        self.creation_time = time.time()
        self.last_update = time.time()

    def calculate_signal_quality(self, data: bytes) -> float:
        """Calcula qualidade do sinal baseado na estrutura dos dados"""
        try:
            if len(data) == 0:
                return 0.0

            # 1. Verificar se dados parecem válidos (não apenas zeros)
            zero_ratio = data.count(b'\x00') / len(data)

            # 2. Verificar diversidade de bytes
            unique_bytes = len(set(data)) / 256

            # 3. Penalizar dados muito repetitivos
            repetition_penalty = 0
            if len(data) > 10:
                if data[:5] * (len(data) // 5) == data[:len(data) - (len(data) % 5)]:
                    repetition_penalty = 0.5

            quality = (1 - zero_ratio) * unique_bytes * (1 - repetition_penalty)
            return max(0.0, min(1.0, quality))

        except Exception:
            return 0.5

    def add_part(self, part_number: int, data: bytes, signal_quality: float = None) -> bool:
        """Adiciona uma parte com verificação de qualidade e retorna True se completo"""
        if 0 <= part_number < self.total_parts:
            if signal_quality is None:
                signal_quality = self.calculate_signal_quality(data)

            if self.parts[part_number] is not None:
                existing_quality = self.parts_quality[part_number]

                if signal_quality > existing_quality:
                    print(
                        f"Substituindo parte {part_number} (qualidade {existing_quality:.3f} -> {signal_quality:.3f})")
                    self.parts[part_number] = data
                    self.parts_quality[part_number] = signal_quality
                    self.last_update = time.time()
                else:
                    print(
                        f"Ignorando parte {part_number} duplicada com qualidade inferior ({signal_quality:.3f} <= {existing_quality:.3f})")
                    return self.received_parts == self.total_parts
            else:
                self.parts[part_number] = data
                self.parts_quality[part_number] = signal_quality
                self.received_parts += 1
                self.last_update = time.time()

            return self.received_parts == self.total_parts
        return False

    def get_progress(self) -> float:
        return (self.received_parts / self.total_parts) * 100 if self.total_parts > 0 else 0

    def get_missing_parts(self) -> list:
        return [i for i, part in enumerate(self.parts) if part is None]

    def assemble_file(self) -> bytes:
        if self.received_parts != self.total_parts:
            missing = self.get_missing_parts()
            raise ValueError(f"Partes insuficientes: {self.received_parts}/{self.total_parts}. Faltando: {missing}")

        complete_data = b''.join(self.parts)

        if len(complete_data) != self.file_size:
            print(f"Aviso: Tamanho do arquivo diferente. Esperado: {self.file_size}, Obtido: {len(complete_data)}")

        actual_crc = binascii.crc32(complete_data) & 0xffffffff
        if actual_crc != self.expected_crc:
            print(f"Aviso: CRC diferente. Esperado: {self.expected_crc:08X}, Obtido: {actual_crc:08X}")

        return complete_data

    def is_expired(self, timeout_seconds: int = 3600) -> bool:
        return (time.time() - self.last_update) > timeout_seconds

    def get_quality_report(self) -> dict:
        return {
            'average_quality': sum(self.parts_quality) / len(self.parts_quality) if self.parts_quality else 0,
            'min_quality': min(self.parts_quality) if self.parts_quality else 0,
            'max_quality': max(self.parts_quality) if self.parts_quality else 0,
            'completed_parts': self.received_parts,
            'total_parts': self.total_parts
        }


class MLSignalProcessor:
    def __init__(self):
        self.anomaly_detector = None
        self.signal_features = []
        self._initialize_detector()

    def _initialize_detector(self):
        """Inicializa detector de anomalias"""
        try:
            self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        except:
            self.anomaly_detector = None

    def extract_features(self, signal: np.ndarray) -> list:
        """Extrai features do sinal para análise ML"""
        if len(signal) < 100:
            return [0] * 10

        features = [
            np.mean(signal),  # Média
            np.std(signal),  # Desvio padrão
            np.max(np.abs(signal)),  # Pico máximo
            np.median(signal),  # Mediana
            np.percentile(signal, 75),  # Quartil superior
            np.percentile(signal, 25),  # Quartil inferior
            len(signal),  # Comprimento
            np.sum(signal > 0.1),  # Amostras acima do threshold
            np.var(signal),  # Variância
            np.mean(np.diff(signal) ** 2)  # Energia da derivada
        ]
        return features

    def detect_anomalies(self, signal: np.ndarray) -> float:
        """Detecta anomalias no sinal usando ML"""
        if self.anomaly_detector is None or len(signal) < 100:
            return 0.0

        features = self.extract_features(signal)
        anomaly_score = self.anomaly_detector.decision_function([features])[0]
        return max(0.0, min(1.0, 1.0 - anomaly_score))


class AdvancedFileAssembly(FileAssembly):
    def __init__(self, filename: str, total_parts: int, file_size: int, file_crc: int):
        super().__init__(filename, total_parts, file_size, file_crc)
        # Adicione funcionalidades avançadas aqui


file_assemblies = {}
reception_stats = {
    'total_files': 0,
    'total_bytes': 0,
    'success_rate': 0.0,
    'last_reception': None,
    'average_quality': 0.0,
    'duplicates_rejected': 0,
    'parts_reordered': 0,
    'total_quality': 0.0,
    'quality_samples': 0
}

RECV_DIR = "recv"
os.makedirs(RECV_DIR, exist_ok=True)


def parse_fbp_stream_enhanced(raw: bytes) -> list:
    parsed = []
    i = 0
    while i < len(raw):
        if raw[i:i+4] == b'\xAA\xAA\xAA\xAA' and raw[i+4:i+8] == b'FBPC':
            fname_len = raw[i+8]
            fname = raw[i+9:i+9+fname_len].decode('utf-8')
            offset = i + 9 + fname_len
            part_number, total_parts, file_size, file_crc, part_size, part_crc, quality_byte = struct.unpack('<IIIIIIB', raw[offset:offset+25])
            payload = raw[offset+25:offset+25+part_size]
            if binascii.crc32(payload) == part_crc:
                is_multi = total_parts > 1
                parsed.append((fname, payload, is_multi, part_number, total_parts, file_size, file_crc))
            i = offset + 25 + part_size
        else:
            i += 1
    return parsed


def smart_decompress(compressed_data: bytes) -> bytes:
    try:
        if compressed_data.startswith(b'LZMA'):
            return lzma.decompress(compressed_data[4:])
        elif compressed_data.startswith(b'DLZM'):
            lzma_decompressed = lzma.decompress(compressed_data[4:])
            return delta_decompress(lzma_decompressed)
        elif compressed_data.startswith(b'ZLIB'):
            return zlib.decompress(compressed_data[4:])
        elif compressed_data.startswith(b'RAW'):
            return compressed_data[4:]
        else:
            try:
                return zlib.decompress(compressed_data)
            except:
                return compressed_data
    except Exception as e:
        print(f"⚠️ Erro na descompressão inteligente: {e}")
        return compressed_data


def save_decoded_files(parsed: list) -> list:
    saved = []
    for fname, payload, is_multi, part_number, total_parts, file_size, file_crc in parsed:
        if is_multi:
            assembly_key = f"{fname}_{file_crc}"
            if assembly_key not in file_assemblies:
                file_assemblies[assembly_key] = AdvancedFileAssembly(fname, total_parts, file_size, file_crc)
            assembly = file_assemblies[assembly_key]
            if assembly.add_part(part_number, payload):
                try:
                    final_data = assembly.assemble_file()
                    final_crc = binascii.crc32(final_data) & 0xffffffff
                    final_size = len(final_data)
                    if final_size != assembly.file_size:
                        print(f"ALERTA: Tamanho do arquivo montado não corresponde! Esperado: {assembly.file_size}, Obtido: {final_size}")
                    if final_crc != assembly.expected_crc:
                        print(f"ALERTA FINAL: CRC do arquivo montado não corresponde! Esperado: {assembly.expected_crc:08X}, Obtido: {final_crc:08X}")
                    timestamp = int(time.time())
                    safe_filename = "".join(c for c in fname if c.isalnum() or c in (' ', '-', '_', '.'))
                    outpath = os.path.join(RECV_DIR, f"recv_{timestamp}_{safe_filename}")
                    with open(outpath, 'wb') as f:
                        f.write(final_data)
                    saved.append(outpath)
                    reception_stats['total_files'] += 1
                    reception_stats['total_bytes'] += len(final_data)
                    reception_stats['last_reception'] = time.time()
                    quality_report = assembly.get_quality_report()
                    print(f"Arquivo multi-partes montado com sucesso: {fname}")
                    print(f"Relatório de qualidade: {quality_report}")
                    del file_assemblies[assembly_key]
                except Exception as e:
                    print(f"Erro ao montar arquivo {fname}: {e}")
            continue

        try:
            final_data = smart_decompress(payload)
            timestamp = int(time.time())
            safe_filename = "".join(c for c in fname if c.isalnum() or c in (' ', '-', '_', '.'))
            outpath = os.path.join(RECV_DIR, f"recv_{timestamp}_{safe_filename}")
            with open(outpath, 'wb') as f:
                f.write(final_data)
            saved.append(outpath)
            reception_stats['total_files'] += 1
            reception_stats['total_bytes'] += len(final_data)
            reception_stats['last_reception'] = time.time()
        except Exception as e:
            print(f"Erro ao salvar arquivo {fname}: {e}")

    current_time = time.time()
    expired_keys = []

    for key, assembly in file_assemblies.items():
        if assembly.is_expired():
            expired_keys.append(key)

    for key in expired_keys:
        assembly = file_assemblies[key]
        print(f"Removendo arquivo incompleto expirado: {assembly.filename} ({assembly.received_parts}/{assembly.total_parts} partes)")
        del file_assemblies[key]

    if parsed:
        reception_stats['success_rate'] = (len(saved) / len(parsed)) * 100

    return saved


def decode_with_retry(data: np.ndarray, mode: str, symbol_rate: int, max_retries: int = 2):
    for attempt in range(max_retries + 1):
        try:
            print(f"Tentativa {attempt + 1} de demodulação no modo {mode}, taxa {symbol_rate}")

            demodulation_map = {
                "FSK1200": lambda: fsk_demodulate(data, baud=1200, mark_freq=1200.0, space_freq=2200.0),
                "FSK9600": lambda: fsk_demodulate(data, baud=9600),
                "BPSK": lambda: bpsk_demodulate(data, baud=symbol_rate, carrier=3000.0),
                "QPSK": lambda: qpsk_demodulate(data, baud=symbol_rate, carrier=3000.0),
                "8PSK": lambda: psk8_demodulate(data, baud=symbol_rate, carrier=12000.0),
                "FSK19200": lambda: fsk_high_speed_demodulate(data, baud=19200),
                "OFDM4": lambda: ofdm_demodulate_simple(data, baud=symbol_rate, carrier=12000.0, num_subcarriers=4),
                "OFDM8": lambda: ofdm_demodulate_simple(data, baud=symbol_rate, carrier=12000.0, num_subcarriers=8),
                "FT8": lambda: ft8_demodulate(data, baud=symbol_rate, carrier=3000.0),
                "PSK31": lambda: psk31_demodulate(data, baud=symbol_rate, carrier=3000.0),
                "FELD_HELL": lambda: feld_hell_demodulate(data, baud=122.5, carrier=1000.0),
            }

            if mode not in demodulation_map:
                print(f"Modo {mode} não encontrado, usando QPSK como fallback")
                raw = qpsk_demodulate(data, baud=symbol_rate, carrier=3000.0)
            else:
                raw = demodulation_map[mode]()

            print(f"Demodulação retornou {len(raw)} bytes")

            # SALVAR RAW PARA INSPEÇÃO
            with open("demodulated.bin", "wb") as f:
                f.write(raw)

            if len(raw) > 0:
                print(f"Primeiros 20 bytes demodulados: {raw[:20].hex()}")
                parsed = parse_fbp_stream_enhanced(raw)
                return save_decoded_files(parsed)
            else:
                print("AVISO: Demodulação retornou 0 bytes")
                return []

        except Exception as e:
            print(f"Erro na tentativa {attempt + 1}: {e}")
            import traceback
            traceback.print_exc()

            if attempt == max_retries:
                print(f"Falha na demodulação após {max_retries + 1} tentativas")
                return []
            else:
                print(f"Tentativa {attempt + 1} falhou, tentando novamente...")


def decode_wav_file(path: str, mode: str = "FSK9600", symbol_rate: int = 9600) -> list:
    try:
        print(f"Decodificando arquivo: {path}")
        data, sr = sf.read(path, always_2d=False)

        if data.ndim > 1:
            data = data.mean(axis=1)

        if sr != SAMPLE_RATE:
            from scipy import signal
            num_samples = int(len(data) * SAMPLE_RATE / sr)
            data = signal.resample(data, num_samples)

        return decode_with_retry(data, mode, symbol_rate)

    except Exception as e:
        print(f"Erro na decodificação do arquivo WAV: {e}")
        return []


def decode_from_buffer(data: np.ndarray, mode: str, symbol_rate: int) -> list:
    return decode_with_retry(data, mode, symbol_rate)


def get_assembly_status():
    status = []
    for assembly in file_assemblies.values():
        status.append({
            'filename': assembly.filename,
            'received': assembly.received_parts,
            'total': assembly.total_parts,
            'progress': assembly.get_progress(),
            'last_update': assembly.last_update,
            'missing_parts': assembly.get_missing_parts()
        })
    return status


def get_reception_stats():
    return reception_stats.copy()


def clear_reception_stats():
    global reception_stats
    reception_stats = {
        'total_files': 0,
        'total_bytes': 0,
        'success_rate': 0.0,
        'last_reception': None,
        'average_quality': 0.0,
        'duplicates_rejected': 0,
        'parts_reordered': 0,
        'total_quality': 0.0,
        'quality_samples': 0
    }