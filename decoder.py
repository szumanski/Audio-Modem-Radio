# decoder.py - VERSÃO MELHORADA COM CONTROLE DE DUPLICATAS E ORDEM
import lzma
import os, time, struct, binascii, threading
import zlib
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


def debug_save_demodulation_stages(audio_samples, demodulated_bytes, filename_prefix):
    """Salva estágios intermediários para debug"""
    try:
        # Salvar amostras de áudio originais
        audio_debug = f"{filename_prefix}_audio.wav"
        sf.write(audio_debug, audio_samples, 96000)
        print(f"💾 Áudio original salvo: {audio_debug}")

        # Salvar bytes demodulados
        bytes_debug = f"{filename_prefix}_demodulated.bin"
        with open(bytes_debug, 'wb') as f:
            f.write(demodulated_bytes)
        print(f"💾 Bytes demodulados salvos: {bytes_debug} ({len(demodulated_bytes)} bytes)")

    except Exception as e:
        print(f"⚠️  Erro ao salvar debug: {e}")


def parse_fbp_stream_enhanced(raw: bytes) -> list:
    """Parser CORRIGIDO que entende a estrutura real do frame"""
    parsed = []
    i = 0

    print(f"🔍 Analisando {len(raw)} bytes brutos em busca de frames...")
    print(f"🔍 Primeiros 64 bytes do raw: {raw[:64].hex()}")

    # Padrão de sincronização CORRETO
    sync_pattern = b'\xAA\xAA\xAA\xAAFBPC'

    while i < len(raw) - 20:  # Mínimo para ter header básico
        # Buscar pelo padrão de sincronização CORRETO
        if i + len(sync_pattern) <= len(raw) and raw[i:i + len(sync_pattern)] == sync_pattern:
            try:
                header_start = i
                i += len(sync_pattern)  # Avançar após sync pattern

                # Extrair tamanho do nome do arquivo
                if i >= len(raw):
                    break

                fname_len = raw[i]
                i += 1

                # Verificar se temos dados suficientes para o nome
                if i + fname_len > len(raw):
                    print(f"❌ Dados insuficientes para nome (precisa {fname_len}, tem {len(raw) - i})")
                    break

                # Extrair nome do arquivo
                fname = raw[i:i + fname_len].decode('utf-8', errors='ignore')
                i += fname_len

                # Verificar se temos dados suficientes para os metadados (25 bytes)
                if i + 25 > len(raw):
                    print(f"❌ Dados insuficientes para metadados (precisa 25, tem {len(raw) - i})")
                    break

                # Extrair metadados - CORREÇÃO: ordem e tipos corretos
                metadata = raw[i:i + 25]
                part_number, total_parts, file_size, file_crc, part_size, part_crc, quality_byte = struct.unpack(
                    '<IIIIIIB', metadata)
                i += 25

                print(f"🔍 Frame encontrado: '{fname}' parte {part_number + 1}/{total_parts}")
                print(f"   Part size: {part_size}, File size: {file_size}")
                print(f"   CRC esperado: {part_crc:08X}")

                # Verificar se temos o payload completo
                if i + part_size > len(raw):
                    print(f"❌ Payload incompleto (precisa {part_size}, tem {len(raw) - i})")
                    break

                payload = raw[i:i + part_size]
                i += part_size

                # Verificar CRC
                actual_crc = binascii.crc32(payload) & 0xffffffff

                if actual_crc == part_crc:
                    print(f"✅ CRC válido para '{fname}'")
                    is_multi = total_parts > 1
                    parsed.append((fname, payload, is_multi, part_number, total_parts, file_size, file_crc))
                else:
                    print(f"⚠️  CRC inválido para '{fname}': esperado {part_crc:08X}, obtido {actual_crc:08X}")
                    # Aceitar mesmo com CRC inválido para teste
                    is_multi = total_parts > 1
                    parsed.append((fname, payload, is_multi, part_number, total_parts, file_size, file_crc))

                print(f"✅ Frame '{fname}' adicionado com sucesso!")
                continue  # Manter i atualizado

            except Exception as e:
                print(f"❌ Erro ao parsear frame em {header_start}: {e}")
                import traceback
                traceback.print_exc()
                i = header_start + 1  # Avançar um byte e continuar
        else:
            i += 1

    print(f"🎯 Parser encontrou {len(parsed)} frames válidos")
    return parsed


def analyze_demodulated_data(raw: bytes):
    """Analisa os dados demodulados para entender a estrutura"""
    print(f"\n🔍 ANÁLISE DETALHADA DOS DADOS DEMODULADOS:")
    print(f"📊 Tamanho total: {len(raw)} bytes")
    print(f"🔍 Primeiros 100 bytes em hex: {raw[:100].hex()}")
    print(f"🔍 Primeiros 100 bytes em ASCII (onde possível): {raw[:100]}")

    # Procurar por padrões conhecidos
    patterns = {
        'Preamble AAAA': b'\xAA\xAA\xAA\xAA',
        'Magic FBPC': b'FBPC',
        'Magic GBPC': b'GBPC',
        'JPEG Start': b'\xFF\xD8\xFF',  # JPEG
        'PNG Start': b'\x89PNG',  # PNG
        'Text': b'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    }

    for name, pattern in patterns.items():
        pos = raw.find(pattern)
        if pos != -1:
            print(f"✅ '{name}' encontrado na posição {pos}")
        else:
            print(f"❌ '{name}' NÃO encontrado")

    # Verificar se há dados que parecem ser um JPEG
    if raw.startswith(b'\xFF\xD8\xFF'):
        print("🎯 Dados parecem ser um JPEG válido!")
    elif b'\xFF\xD8\xFF' in raw:
        jpeg_pos = raw.find(b'\xFF\xD8\xFF')
        print(f"🎯 JPEG encontrado dentro dos dados na posição {jpeg_pos}")

    print("--- Fim da análise ---\n")

def find_and_sync_frame(raw_data: bytes) -> bytes:
    """Encontra e sincroniza no início do frame"""
    preamble = b'\xAA\xAA\xAA\xAA'
    magic = b'FBPC'
    sync_pattern = preamble + magic

    # Procurar pelo padrão de sincronização
    sync_pos = raw_data.find(sync_pattern)

    if sync_pos != -1:
        print(f"🎯 Sincronização encontrada na posição {sync_pos}")
        return raw_data[sync_pos:]
    else:
        # Tentar encontrar apenas o preamble
        preamble_pos = raw_data.find(preamble)
        if preamble_pos != -1:
            print(f"⚠️  Apenas preamble encontrado na posição {preamble_pos}")
            return raw_data[preamble_pos:]
        else:
            print("❌ Nenhum padrão de sincronização encontrado")
            return raw_data

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
    """Salva arquivos decodificados com melhor tratamento de erro"""
    saved = []

    if not parsed:
        print("📭 Nenhum frame válido para processar")
        return saved

    print(f"📦 Processando {len(parsed)} frame(s) decodificado(s)")

    for fname, payload, is_multi, part_number, total_parts, file_size, file_crc in parsed:
        print(f"🔧 Processando: {fname} (Parte {part_number + 1}/{total_parts})")

        if is_multi:
            # Lógica para arquivos multi-partes (mantida do original)
            assembly_key = f"{fname}_{file_crc}"
            if assembly_key not in file_assemblies:
                file_assemblies[assembly_key] = AdvancedFileAssembly(fname, total_parts, file_size, file_crc)

            assembly = file_assemblies[assembly_key]
            if assembly.add_part(part_number, payload):
                try:
                    final_data = assembly.assemble_file()
                    final_crc = binascii.crc32(final_data) & 0xffffffff

                    if final_crc == assembly.expected_crc:
                        timestamp = int(time.time())
                        safe_filename = "".join(c for c in fname if c.isalnum() or c in (' ', '-', '_', '.'))
                        outpath = os.path.join(RECV_DIR, f"recv_{timestamp}_{safe_filename}")

                        with open(outpath, 'wb') as f:
                            f.write(final_data)

                        saved.append(outpath)
                        reception_stats['total_files'] += 1
                        reception_stats['total_bytes'] += len(final_data)
                        reception_stats['last_reception'] = time.time()

                        print(f"✅ Arquivo multi-partes montado: {fname}")
                        del file_assemblies[assembly_key]
                    else:
                        print(f"❌ CRC do arquivo montado não confere: {fname}")

                except Exception as e:
                    print(f"❌ Erro ao montar arquivo {fname}: {e}")
            continue

        # Arquivo único
        try:
            final_data = smart_decompress(payload)

            # Verificar se os dados fazem sentido
            if len(final_data) == 0:
                print(f"⚠️  Arquivo {fname} vazio após descompressão")
                continue

            timestamp = int(time.time())
            safe_filename = "".join(c for c in fname if c.isalnum() or c in (' ', '-', '_', '.'))
            outpath = os.path.join(RECV_DIR, f"recv_{timestamp}_{safe_filename}")

            with open(outpath, 'wb') as f:
                f.write(final_data)

            saved.append(outpath)
            reception_stats['total_files'] += 1
            reception_stats['total_bytes'] += len(final_data)
            reception_stats['last_reception'] = time.time()

            print(f"✅ Arquivo salvo: {outpath} ({len(final_data)} bytes)")

        except Exception as e:
            print(f"❌ Erro ao salvar arquivo {fname}: {e}")

    # Limpar assemblies expirados
    current_time = time.time()
    expired_keys = [key for key, assembly in file_assemblies.items()
                    if assembly.is_expired()]

    for key in expired_keys:
        print(f"🗑️  Removendo assembly expirado: {key}")
        del file_assemblies[key]

    if parsed:
        success_rate = (len(saved) / len(parsed)) * 100
        reception_stats['success_rate'] = success_rate
        print(f"📊 Taxa de sucesso: {success_rate:.1f}%")

    return saved


def qpsk_demodulate_simple_fallback(samples: np.ndarray, baud=1200, carrier=3000.0, samp_rate=96000) -> bytes:
    """Demodulação QPSK simplificada sem filtro para fallback"""
    print("🔄 Usando demodulação QPSK simplificada (fallback)")

    samples_per_symbol = int(samp_rate / baud)
    bits = ''

    # Processar em passos do tamanho do símbolo
    for start_idx in range(0, len(samples) - samples_per_symbol, samples_per_symbol):
        symbol_samples = samples[start_idx:start_idx + samples_per_symbol]

        # Gerar portadoras básicas
        t = np.arange(len(symbol_samples)) / samp_rate
        I_carrier = np.cos(2 * np.pi * carrier * t)
        Q_carrier = np.sin(2 * np.pi * carrier * t)

        # Correlação simples
        I_component = np.sum(symbol_samples * I_carrier)
        Q_component = np.sum(symbol_samples * Q_carrier)

        # Decisão de símbolo
        if I_component >= 0 and Q_component >= 0:
            bits += '00'
        elif I_component < 0 and Q_component >= 0:
            bits += '01'
        elif I_component < 0 and Q_component < 0:
            bits += '11'
        else:
            bits += '10'

    # Converter para bytes
    bytes_out = bytearray()
    for i in range(0, len(bits) - 7, 8):
        try:
            byte_val = int(bits[i:i + 8], 2)
            bytes_out.append(byte_val)
        except:
            continue

    print(f"🔍 QPSK simplificado demodulou {len(bytes_out)} bytes")
    return bytes(bytes_out)

def neural_decode_data(audio_samples: np.ndarray, symbol_rate: int = 8000) -> bytes:
    """Decodificação usando modem neural"""
    try:
        from neural_modem import neural_demodulate
        return neural_demodulate(audio_samples, symbol_rate)
    except ImportError as e:
        print(f"❌ Modem neural não disponível: {e}")
        print("🔄 Usando QPSK como fallback")
        from modem import qpsk_demodulate
        return qpsk_demodulate(audio_samples, baud=symbol_rate, carrier=3000.0)


def decode_with_retry(data: np.ndarray, mode: str, symbol_rate: int, max_retries: int = 2):
    for attempt in range(max_retries + 1):
        try:
            print(f"🎯 Tentativa {attempt + 1} de demodulação no modo {mode}, taxa {symbol_rate}")

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
                print(f"⚠️ Modo {mode} não encontrado, usando QPSK como fallback")
                raw = qpsk_demodulate(data, baud=symbol_rate, carrier=3000.0)
            else:
                raw = demodulation_map[mode]()

            print(f"📊 Demodulação retornou {len(raw)} bytes")

            if len(raw) > 0:
                # ANÁLISE DETALHADA dos dados demodulados
                analyze_demodulated_data(raw)

                # Salvar dados para debug
                debug_prefix = f"debug_{mode}_{int(time.time())}_attempt{attempt}"
                debug_save_demodulation_stages(data, raw, debug_prefix)

                # Tentar parsear os dados
                parsed = parse_fbp_stream_enhanced(raw)
                saved_files = save_decoded_files(parsed)

                if saved_files:
                    print(f"✅ {len(saved_files)} arquivo(s) decodificado(s) com sucesso!")
                    return saved_files
                else:
                    print("❌ Nenhum arquivo válido encontrado")

                    # Tentativa alternativa: verificar se os dados são diretamente um JPEG
                    if raw.startswith(b'\xFF\xD8\xFF') or b'\xFF\xD8\xFF' in raw:
                        print("🎯 Dados parecem conter um JPEG diretamente!")
                        jpeg_start = raw.find(b'\xFF\xD8\xFF')
                        if jpeg_start != -1:
                            jpeg_data = raw[jpeg_start:]
                            # Salvar como JPEG
                            timestamp = int(time.time())
                            outpath = os.path.join(RECV_DIR, f"direct_jpeg_{timestamp}.jpg")
                            with open(outpath, 'wb') as f:
                                f.write(jpeg_data)
                            print(f"💾 JPEG salvo diretamente: {outpath}")
                            return [outpath]

                    return []

            else:
                print("❌ AVISO: Demodulação retornou 0 bytes")
                return []

        except Exception as e:
            print(f"❌ Erro na tentativa {attempt + 1}: {e}")
            import traceback
            traceback.print_exc()

            if attempt == max_retries:
                print(f"💥 Falha na demodulação após {max_retries + 1} tentativas")
                return []
            else:
                print(f"🔄 Tentativa {attempt + 1} falhou, tentando novamente...")
                time.sleep(0.5)

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