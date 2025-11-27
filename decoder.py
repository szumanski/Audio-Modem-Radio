import lzma
import os, time, struct, binascii, threading
import zlib
from collections import defaultdict, deque
from datetime import datetime
import sounddevice as sd
import numpy as np
import soundfile as sf
import scipy.signal as signal
from typing import Dict
from collections import defaultdict
from modem import (fsk_demodulate, bpsk_demodulate, qpsk_demodulate,
                   psk8_demodulate, fsk_high_speed_demodulate, ofdm_demodulate_simple,
                   SAMPLE_RATE, ft8_demodulate, psk31_demodulate, feld_hell_demodulate)
from utils.compression import decompress_data, super_decompress, delta_decompress, intelligent_decompress

RECV_DIR = "recv"
os.makedirs(RECV_DIR, exist_ok=True)
active_file_assemblies: Dict[str, 'FileAssembly'] = {}
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
    """Parser robusto que procura Magic Bytes em qualquer lugar"""
    parsed_files = []

    # Magic bytes FBPC
    magic = b'FBPC'

    # Busca ingênua por todos os índices que contêm o Magic
    # Nota: O Modem agora tenta alinhar os bits para que os bytes saiam alinhados
    # Se o modem falhar no bit-alignment, este find falhará.

    start_indices = []
    offset = 0
    while True:
        idx = raw.find(magic, offset)
        if idx == -1: break
        start_indices.append(idx)
        offset = idx + 1

    print(f"Encontrados {len(start_indices)} candidatos a cabeçalho.")

    for start in start_indices:
        try:
            # Header minimo: Magic(4) + Len(1) + Name(1) + Meta(20)
            if start + 30 > len(raw): continue

            # Ler tamanho do nome
            name_len = raw[start + 4]
            if name_len == 0: continue

            # Ler nome
            name_start = start + 5
            fname = raw[name_start: name_start + name_len].decode('utf-8', 'ignore')

            # Metadados
            meta_start = name_start + name_len
            # Part(4) + Total(4) + FSize(4) + FCrc(4) + DataLen(4) + PartCrc(4)
            if meta_start + 24 > len(raw): continue

            (part_num, total_parts, fsize, fcrc, dlen, pcrc) = struct.unpack('<IIIIII', raw[meta_start:meta_start + 24])

            # Validar sanidade
            if dlen > 50_000_000 or dlen == 0: continue  # Tamanho absurdo

            payload_start = meta_start + 24
            if payload_start + dlen > len(raw):
                print(f"Dados incompletos para {fname}")
                continue

            payload = raw[payload_start: payload_start + dlen]

            # Validar CRC
            calc_crc = binascii.crc32(payload) & 0xffffffff
            if calc_crc == pcrc:
                print(f"✅ CRC VÁLIDO: {fname} (Parte {part_num + 1}/{total_parts})")
                parsed_files.append({
                    'name': fname,
                    'data': payload,
                    'final_crc': fcrc
                })
            else:
                print(f"❌ Erro de CRC para {fname}")

        except Exception as e:
            print(f"Erro no parse candidato {start}: {e}")

    return parsed_files

def smart_decompress(compressed_data: bytes) -> bytes:
    """Descompressão inteligente com fallbacks"""
    try:
        print(f"🔍 Tentando descomprimir {len(compressed_data)} bytes...")

        if compressed_data.startswith(b'LZMA'):
            print("📦 Usando descompressão LZMA")
            import lzma
            return lzma.decompress(compressed_data[4:])
        elif compressed_data.startswith(b'DLZM'):
            print("📦 Usando descompressão Delta+LZMA")
            import lzma
            lzma_decompressed = lzma.decompress(compressed_data[4:])
            return delta_decompress(lzma_decompressed)
        elif compressed_data.startswith(b'ZLIB'):
            print("📦 Usando descompressão ZLIB")
            import zlib
            return zlib.decompress(compressed_data[4:])
        elif compressed_data.startswith(b'RAW'):
            print("📦 Dados RAW, sem compressão")
            return compressed_data[4:]
        else:
            # Tentar descompressão automática
            try:
                print("🔍 Tentando descompressão ZLIB automática...")
                import zlib
                return zlib.decompress(compressed_data)
            except:
                print("📦 Dados não comprimidos, retornando raw")
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


def decode_with_retry(data: np.ndarray, mode: str, symbol_rate: int, max_retries: int = 3):
    """Decodificação com múltiplas tentativas e parâmetros ajustáveis"""

    best_result = []
    best_quality = 0

    for attempt in range(max_retries):
        try:
            print(f"🎯 Tentativa {attempt + 1} de demodulação no modo {mode}, taxa {symbol_rate}")

            # Ajustar parâmetros baseado na tentativa
            if attempt == 1:
                symbol_rate = int(symbol_rate * 0.95)  # Reduzir slightly na segunda tentativa
            elif attempt == 2:
                symbol_rate = int(symbol_rate * 1.05)  # Aumentar slightly na terceira tentativa

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

            if len(raw) > 100:  # Reduzir limite mínimo
                # SALVAR RAW PARA INSPEÇÃO (apenas se for significativo)
                with open(f"demodulated_attempt_{attempt}.bin", "wb") as f:
                    f.write(raw)

                print(f"Primeiros 50 bytes demodulados: {raw[:50].hex()}")
                parsed = parse_fbp_stream_enhanced(raw)

                if parsed:
                    saved_files = save_decoded_files(parsed)
                    if saved_files:
                        print(f"✅ Tentativa {attempt + 1} bem-sucedida: {len(saved_files)} arquivos salvos")
                        return saved_files
                    else:
                        print(f"⚠️ Tentativa {attempt + 1}: frames encontrados mas não salvos")
                else:
                    print(f"⚠️ Tentativa {attempt + 1}: nenhum frame encontrado")
            else:
                print(f"⚠️ Tentativa {attempt + 1}: demodulação retornou apenas {len(raw)} bytes")

        except Exception as e:
            print(f"❌ Erro na tentativa {attempt + 1}: {e}")
            import traceback
            traceback.print_exc()

    print(f"❌ Falha na demodulação após {max_retries} tentativas")
    return []


def decode_wav_file(path: str, mode: str, symbol_rate: int) -> list:
    data, sr = sf.read(path)
    if len(data.shape) > 1: data = data[:, 0]  # Mono

    # Resample se necessario
    if sr != SAMPLE_RATE:
        number_of_samples = int(round(len(data) * float(SAMPLE_RATE) / sr))
        data = signal.resample(data, number_of_samples)

    return decode_from_buffer(data, mode, symbol_rate)


def calculate_global_average_quality() -> float:
    """Calcula a qualidade média do sinal recebido em todos os arquivos ativos."""
    global active_file_assemblies
    total_quality = 0.0
    total_parts = 0

    # Safeguard: If not initialized, return 0.0 and log a warning
    try:
        if not active_file_assemblies:
            return 0.0
    except NameError:
        print("Warning: active_file_assemblies not defined. Initializing now.")
        active_file_assemblies = {}  # Sem 'global' aqui, pois já declarado no topo
        return 0.0

    for assembly in active_file_assemblies.values():
        if assembly.parts_quality and assembly.received_parts > 0:
            # Pondera pela qualidade das partes realmente recebidas
            # Filtra partes com qualidade 0.0 que podem não ter sido recebidas ou iniciadas
            qualities = [q for q in assembly.parts_quality if q > 0]
            total_quality += sum(qualities)
            total_parts += len(qualities)

    return (total_quality / total_parts) if total_parts > 0 else 0.0

def decode_from_buffer(data: np.ndarray, mode: str, symbol_rate: int) -> list:
    print(f"Demodulando {len(data)} amostras em modo {mode}...")

    try:
        raw_bytes = b''
        if mode == "BPSK":
            raw_bytes = bpsk_demodulate(data, baud=symbol_rate)
        elif mode == "QPSK" or mode == "8PSK":
            raw_bytes = qpsk_demodulate(data, baud=symbol_rate)
        elif mode.startswith("FSK"):
            baud = 1200
            if "9600" in mode:
                baud = 9600
            elif "19200" in mode:
                baud = 19200
            raw_bytes = fsk_demodulate(data, baud=baud)
        else:
            raw_bytes = qpsk_demodulate(data, baud=symbol_rate)

        print(f"Bytes brutos demodulados: {len(raw_bytes)}")
        # Dump para debug se precisar
        # with open("last_demod.bin", "wb") as f: f.write(raw_bytes)

        frames = parse_fbp_stream_enhanced(raw_bytes)
        saved = []

        for frame in frames:
            try:
                # Descompressão
                final_data = intelligent_decompress(frame['data'])

                ts = int(time.time())
                clean_name = os.path.basename(frame['name'])
                path = os.path.join(RECV_DIR, f"{ts}_{clean_name}")

                with open(path, 'wb') as f:
                    f.write(final_data)
                saved.append(path)
            except Exception as e:
                print(f"Erro salvando arquivo: {e}")

        return saved

    except Exception as e:
        print(f"Erro crítico na demodulação: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_assembly_status(): return []


def find_frame_start(data: bytes, start_pos: int = 0) -> int:
    """Encontra o início do próximo frame de forma robusta"""
    preamble = b'\xAA\xAA\xAA\xAA'
    magic = b'FBPC'

    for i in range(start_pos, len(data) - 8):
        if data[i:i + 4] == preamble and data[i + 4:i + 8] == magic:
            return i
    return -1


def get_reception_stats():
    global reception_stats
    stats = reception_stats.copy()

    # 💥 CORREÇÃO PRINCIPAL: Garante que 'average_quality' esteja sempre presente
    stats['average_quality'] = calculate_global_average_quality()

    return stats

def debug_demodulation(samples: np.ndarray, mode: str, symbol_rate: int):
    """Função para debug da demodulação"""
    print(f"🔍 DEBUG Demodulação:")
    print(f"   - Modo: {mode}")
    print(f"   - Taxa: {symbol_rate}")
    print(f"   - Amostras: {len(samples)}")
    print(f"   - Primeiras 20 amostras: {samples[:20]}")
    print(f"   - Média: {np.mean(samples):.6f}")
    print(f"   - Std: {np.std(samples):.6f}")
    print(f"   - Min/Max: {np.min(samples):.6f}/{np.max(samples):.6f}")

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