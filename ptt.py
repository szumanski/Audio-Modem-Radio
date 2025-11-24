# ptt.py
import time
import serial
import serial.tools.list_ports


class PTTManager:
    def __init__(self):
        self.ser = None
        self.port = None
        self.method = "RTS"  # Pode ser 'RTS' ou 'DTR'

    @staticmethod
    def get_available_ports():
        """Retorna lista de portas COM disponíveis"""
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]

    def connect(self, port_name, method="RTS"):
        self.port = port_name
        self.method = method

    def ptt_on(self):
        """Ativa o PTT"""
        if not self.port or self.port == "Nenhuma":
            return

        try:
            if self.ser is None or not self.ser.is_open:
                self.ser = serial.Serial(self.port, 9600, timeout=1)

            # A maioria das interfaces usa RTS para PTT, algumas usam DTR
            if self.method == "RTS":
                self.ser.rts = True
                self.ser.dtr = False
            else:
                self.ser.dtr = True
                self.ser.rts = False

            print(f"🔴 PTT ON ({self.port} via {self.method})")
            time.sleep(0.2)  # Pequeno delay para o rádio armar (Pre-TX Delay)
        except Exception as e:
            print(f"Erro ao ativar PTT: {e}")

    def ptt_off(self):
        """Desativa o PTT"""
        if self.ser and self.ser.is_open:
            try:
                self.ser.rts = False
                self.ser.dtr = False
                self.ser.close()
                self.ser = None
                print(f"⚪ PTT OFF")
            except Exception as e:
                print(f"Erro ao desativar PTT: {e}")


# Instância global
ptt_controller = PTTManager()