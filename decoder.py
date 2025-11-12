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
                   SAMPLE_RATE)
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
        self.ml_processor = MLSignalProcessor()
        self.reception_timestamps = []
        self.signal_quality_history = []

    def calculate_signal_quality_ml(self, data: bytes, full_signal: np.ndarray = None) -> float:
        """Calcula qualidade do sinal usando machine learning"""
        base_quality = self.calculate_signal_quality(data)

        if full_signal is not None and len(full_signal) > 100:
            anomaly_score = self.ml_processor.detect_anomalies(full_signal)
            # Combinar qualidade base com score de anomalia
            ml_quality = base_quality * (1.0 - anomaly_score * 0.5)
            return max(0.0, min(1.0, ml_quality))

        return base_quality

    def add_part_with_ml(self, part_number: int, data: bytes,
                         signal_quality: float = None,
                         raw_signal: np.ndarray = None) -> bool:
        """Adiciona parte com análise ML avançada"""
        if raw_signal is not None and signal_quality is None:
            signal_quality = self.calculate_signal_quality_ml(data, raw_signal)

        self.reception_timestamps.append(time.time())
        self.signal_quality_history.append(signal_quality)

        return self.add_part(part_number, data, signal_quality)


# Diretório de recepção com organização por data
def get_received_dir():
    date_str = datetime.now().strftime("%Y-%m-%d")
    recv_dir = os.path.join("received", date_str)
    os.makedirs(recv_dir, exist_ok=True)
    return recv_dir


RECV_DIR = get_received_dir()

# Sistema global para rastrear arquivos multi-partes
file_assemblies = {}  # Mudado de defaultdict(dict) para dict normal
reception_stats = {
    'total_files': 0,
    'total_bytes': 0,
    'success_rate': 0.0,
    'last_reception': None,
    'average_quality': 0.0,
    'duplicates_rejected': 0,
    'parts_reordered': 0,
    'total_quality': 0.0,  # Adicionado
    'quality_samples': 0  # Adicionado
}


def parse_fbp_stream_enhanced(b: bytes):
    """Parser melhorado com diagnóstico detalhado"""
    i = 0
    res = []
    L = len(b)
    frames_found = 0
    errors = 0

    print(f"🔍 Iniciando parsing de {L} bytes...")

    while i < L - 25:
        idx = b.find(b'\xAA\xAA\xAA\xAAFBPC', i)
        if idx == -1:
            if frames_found == 0 and L > 50:
                print(f"Primeiros 50 bytes: {b[:50].hex()}")
            break

        print(f"✅ Preamble encontrado na posição {idx}")

        try:
            j = idx + 8
            if j >= L:
                break

            fname_len = b[j]
            j += 1
            print(f"📁 Tamanho do nome do arquivo: {fname_len}")

            if j + fname_len > L:
                i = idx + 1
                continue

            fname = b[j:j + fname_len].decode('utf-8', errors='ignore')
            j += fname_len
            print(f"📄 Nome do arquivo: {fname}")

            if j + 25 > L:
                i = idx + 1
                continue

            part_number = struct.unpack('<I', b[j:j + 4])[0]
            total_parts = struct.unpack('<I', b[j + 4:j + 8])[0]
            file_size = struct.unpack('<I', b[j + 8:j + 12])[0]
            file_crc = struct.unpack('<I', b[j + 12:j + 16])[0]
            part_size = struct.unpack('<I', b[j + 16:j + 20])[0]
            part_crc = struct.unpack('<I', b[j + 20:j + 24])[0]
            quality_byte = b[j + 24]
            signal_quality = quality_byte / 255.0

            j += 25
            print(f"🔢 Part {part_number + 1}/{total_parts}, Tamanho: {part_size} bytes")

            if j + part_size > L:
                i = idx + 1
                continue

            payload = b[j:j + part_size]

            calculated_crc = binascii.crc32(payload) & 0xffffffff
            if calculated_crc == part_crc:
                print(f"✅ CRC válido para parte {part_number + 1}")
                res.append((fname, payload, part_number, total_parts, file_size, file_crc, signal_quality))
                frames_found += 1
                i = j + part_size
            else:
                print(f"❌ CRC inválido: esperado {part_crc:08X}, calculado {calculated_crc:08X}")
                i = idx + 1

        except Exception as e:
            print(f"❌ Erro no parsing do frame: {e}")
            i = idx + 1

    print(f"📊 Frames encontrados: {frames_found}, Erros: {errors}")
    return res


def parse_fbp_stream_ml_enhanced(b: bytes, raw_signal: np.ndarray = None):
    """Parser com machine learning para melhor detecção"""
    ml_processor = MLSignalProcessor()
    parsed_frames = parse_fbp_stream_enhanced(b)

    # Analisar qualidade dos frames com ML se sinal bruto disponível
    if raw_signal is not None:
        for i, frame in enumerate(parsed_frames):
            fname, payload, part_number, total_parts, file_size, file_crc, signal_quality = frame
            anomaly_score = ml_processor.detect_anomalies(raw_signal)
            adjusted_quality = signal_quality * (1.0 - anomaly_score * 0.3)
            parsed_frames[i] = (fname, payload, part_number, total_parts,
                                file_size, file_crc, adjusted_quality)

    return parsed_frames


def calculate_payload_quality(payload: bytes, full_frame: bytes) -> float:
    try:
        quality_metrics = []

        if len(payload) > 0:
            non_null = sum(1 for byte in payload if byte != 0)
            data_ratio = non_null / len(payload)
            quality_metrics.append(data_ratio)

        if len(full_frame) > 20:
            preamble_quality = 1.0 if full_frame[:8] == b'\xAA\xAA\xAA\xAAFBPC' else 0.5
            quality_metrics.append(preamble_quality)

        expected_size = struct.unpack('<I', full_frame[16:20])[0] if len(full_frame) >= 20 else 0
        actual_size = len(payload)
        size_match = 1.0 if expected_size == actual_size else 0.5
        quality_metrics.append(size_match)

        return sum(quality_metrics) / len(quality_metrics) if quality_metrics else 0.5

    except Exception:
        return 0.5


def smart_decompress(payload: bytes) -> bytes:
    """
    Descompressão inteligente com a ordem de operações corrigida.
    Primeiro desfaz o delta, depois o super_decompress.
    """
    try:
        # Passo 1: Tenta reverter a compressão delta primeiro.
        # É seguro aplicar mesmo que não tenha sido usado, pois o impacto é mínimo em dados não-delta.
        try:
            payload_after_delta = delta_decompress(payload)
        except Exception:
            # Se a descompressão delta falhar, usa o payload original
            payload_after_delta = payload

        # Passo 2: Agora, com os prefixos restaurados, faz a super descompressão.
        if payload_after_delta.startswith((b'LZMA', b'ZLIB', b'RAW')):
            return super_decompress(payload_after_delta)

        # Passo 3: Se não for um formato super_compress, tenta zlib padrão.
        # Isso cobre os modos de baixa velocidade e casos onde a compressão delta não foi aplicada.
        # Usamos o payload original aqui, pois a compressão delta não se aplicaria.
        return decompress_data(payload)

    except Exception as e:
        print(f"Erro severo na descompressão, retornando dados brutos: {e}")
        # Como último recurso, retorna o payload original.
        return payload


def save_decoded_files(parsed):
    saved = []
    global reception_stats

    for fname, payload, part_number, total_parts, file_size, file_crc, signal_quality in parsed:
        # Atualizar métricas de qualidade
        reception_stats['total_quality'] += signal_quality
        reception_stats['quality_samples'] += 1
        reception_stats['average_quality'] = (reception_stats['total_quality'] / reception_stats[
            'quality_samples']) * 100

        if total_parts > 1:
            assembly_key = f"{os.path.basename(fname).lower()}_{file_crc:08x}"

            if assembly_key not in file_assemblies:
                file_assemblies[assembly_key] = FileAssembly(fname, total_parts, file_size, file_crc)
                print(f"Iniciando montagem do arquivo: {fname} ({total_parts} partes)")

            assembly = file_assemblies[assembly_key]
            is_complete = assembly.add_part(part_number, payload, signal_quality)

            progress = assembly.get_progress()
            missing = assembly.get_missing_parts()

            print(
                f"Parte {part_number + 1}/{total_parts} de {fname} recebida ({assembly.received_parts}/{total_parts} completo, {progress:.1f}%)")

            if missing:
                print(f"Partes faltantes de {fname}: {[p + 1 for p in missing]}")
            else:
                print(f"Todas as partes de {fname} recebidas!")

            if is_complete:
                try:
                    print("Todas as partes recebidas. Descomprimindo cada parte individualmente...")

                    # 1. Descomprime cada parte da lista e armazena os resultados.
                    decompressed_parts = [smart_decompress(part) for part in assembly.parts]

                    # 2. Junta os dados já descomprimidos para formar o arquivo final.
                    final_data = b''.join(decompressed_parts)

                    # 3. Agora, a verificação final de tamanho e CRC fará sentido.
                    final_size = len(final_data)
                    final_crc = binascii.crc32(final_data) & 0xffffffff

                    if final_size != assembly.file_size:
                        print(
                            f"ALERTA FINAL: Tamanho do arquivo montado não corresponde! Esperado: {assembly.file_size}, Obtido: {final_size}")

                    if final_crc != assembly.expected_crc:
                        print(
                            f"ALERTA FINAL: CRC do arquivo montado não corresponde! Esperado: {assembly.expected_crc:08X}, Obtido: {final_crc:08X}")

                    timestamp = int(time.time())

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
        print(
            f"Removendo arquivo incompleto expirado: {assembly.filename} ({assembly.received_parts}/{assembly.total_parts} partes)")
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