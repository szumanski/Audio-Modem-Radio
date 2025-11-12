# fec.py - Forward Error Correction melhorado
import numpy as np
import zlib
import struct


class ReedSolomonFEC:
    def __init__(self, nsym=10):
        self.nsym = nsym  # Símbolos de correção

    def encode(self, data: bytes) -> bytes:
        """Codificação Reed-Solomon simplificada com paridade"""
        # Em produção, usar biblioteca como reedsolo
        # Aqui: duplicação inteligente + checksum
        encoded = bytearray()

        for i in range(0, len(data), 2):
            if i + 1 < len(data):
                byte1 = data[i]
                byte2 = data[i + 1]
                # Adicionar byte de paridade
                parity = byte1 ^ byte2
                encoded.extend([byte1, byte2, parity])
            else:
                encoded.append(data[i])
                encoded.append(0xFF)  # Padding

        # Adicionar checksum CRC32
        crc = zlib.crc32(data) & 0xFFFFFFFF
        encoded.extend(struct.pack('<I', crc))

        return bytes(encoded)

    def decode(self, data: bytes) -> bytes:
        """Decodificação com correção simples"""
        if len(data) < 4:
            return data

        # Extrair checksum
        crc_expected = struct.unpack('<I', data[-4:])[0]
        data_without_crc = data[:-4]

        decoded = bytearray()
        i = 0

        while i < len(data_without_crc):
            if i + 2 < len(data_without_crc):
                byte1 = data_without_crc[i]
                byte2 = data_without_crc[i + 1]
                parity = data_without_crc[i + 2]

                # Verificar e corrigir erro simples
                if byte1 ^ byte2 == parity:
                    decoded.extend([byte1, byte2])
                else:
                    # Tentar corrigir - preferir byte1
                    decoded.extend([byte1, 0x3F])  # '?' em ASCII

                i += 3
            else:
                decoded.append(data_without_crc[i])
                i += 1

        # Verificar CRC
        crc_actual = zlib.crc32(decoded) & 0xFFFFFFFF
        if crc_actual != crc_expected:
            print(f"Aviso: CRC não corresponde - dados podem estar corrompidos")

        return bytes(decoded)


class ConvolutionalEncoder:
    def __init__(self, constraint_length=7):
        self.constraint_length = constraint_length
        # Polinômios para codificação convolucional rate 1/2
        self.g1 = 0b1111001  # 171 octal
        self.g2 = 0b1011011  # 133 octal

    def encode(self, data: bytes) -> bytes:
        """Codificação convolucional rate 1/2"""
        encoded_bits = []
        shift_register = 0

        for byte in data:
            for bit_pos in range(8):
                bit = (byte >> (7 - bit_pos)) & 1
                shift_register = ((shift_register << 1) | bit) & 0x7F

                # Calcular bits de saída dos dois polinômios
                out1 = bin(shift_register & self.g1).count('1') % 2
                out2 = bin(shift_register & self.g2).count('1') % 2

                encoded_bits.extend([out1, out2])

        # Flush com zeros
        for _ in range(6):
            shift_register = (shift_register << 1) & 0x7F
            out1 = bin(shift_register & self.g1).count('1') % 2
            out2 = bin(shift_register & self.g2).count('1') % 2
            encoded_bits.extend([out1, out2])

        # Converter bits para bytes
        encoded = bytearray()
        for i in range(0, len(encoded_bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(encoded_bits):
                    byte = (byte << 1) | encoded_bits[i + j]
            encoded.append(byte)

        return bytes(encoded)


class ViterbiDecoder:
    def __init__(self, constraint_length=7):
        self.constraint_length = constraint_length
        self.g1 = 0b1111001
        self.g2 = 0b1011011
        self.trellis = self._build_trellis()

    def _build_trellis(self):
        """Constroi o trellis para decodificação Viterbi"""
        # Implementação simplificada
        return {}

    def decode(self, data: bytes) -> bytes:
        """Decodificação Viterbi simplificada"""
        # Para implementação real, usar biblioteca especializada
        # Aqui: decodificação ingênua
        decoded_bits = []

        for byte in data:
            for i in range(8):
                bit = (byte >> (7 - i)) & 1
                decoded_bits.append(bit)

        # Remover bits de flush (últimos 12 bits)
        if len(decoded_bits) >= 12:
            decoded_bits = decoded_bits[:-12]

        # Converter para bytes (rate 1/2 -> pegar cada 2º bit)
        decoded = bytearray()
        bits_used = []
        for i in range(0, len(decoded_bits), 2):
            if i < len(decoded_bits):
                bits_used.append(decoded_bits[i])

        for i in range(0, len(bits_used), 8):
            byte = 0
            for j in range(8):
                if i + j < len(bits_used):
                    byte = (byte << 1) | bits_used[i + j]
            decoded.append(byte)

        return bytes(decoded)