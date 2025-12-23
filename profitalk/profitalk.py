import socket
import struct
import msgpack
import threading
import multiprocessing
import queue
import requests
import numpy as np
from typing import List, Dict, Any, Optional

class ProfiTalkError(Exception):
    """Custom exception for ProfiTalk errors."""
    pass

class ProfiTalkClient:
    """Client for interacting with RF62x devices using ProfiTalk protocol."""

    DEFAULT_SEARCH_PORT = 51000
    DEFAULT_COMMANDS_PORT = 51001
    DEFAULT_PROFILES_PORT = 51002
    DEFAULT_VIDEO_PORT = 51003

    def __init__(self, ip: str, commands_port: int = DEFAULT_COMMANDS_PORT,
                 profiles_port: int = DEFAULT_PROFILES_PORT,
                 video_port: int = DEFAULT_VIDEO_PORT):
        """
        Initialize the client.

        :param ip: IP address of the device.
        :param commands_port: Port for commands service.
        :param profiles_port: Port for profiles stream.
        :param video_port: Port for video stream.
        """
        self.ip = ip
        self.commands_port = commands_port
        self.profiles_port = profiles_port
        self.video_port = video_port    

    @staticmethod
    def discover_scanners(timeout: float = 1.0, serial: Optional[int] = None, name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Discover scanners in the network using UDP broadcast.

        :param timeout: Timeout for receiving responses.
        :param serial: Optional serial number to filter.
        :param name: Optional name to filter.
        :return: List of discovered scanners' info.
        """
        request = {"request": "SEARCH"}
        if serial is not None:
            request["serial"] = serial
        if name is not None:
            request["name"] = name

        packed = msgpack.packb(request)

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(timeout)

        # Send to broadcast addresses
        broadcast_addresses = ['255.255.255.255']  # Add subnet broadcast if known

        for bcast in broadcast_addresses:
            sock.sendto(packed, (bcast, ProfiTalkClient.DEFAULT_SEARCH_PORT))

        scanners = []
        try:
            while True:
                data, addr = sock.recvfrom(4096)
                response = msgpack.unpackb(data)
                scanners.append(response)
        except socket.timeout:
            pass
        finally:
            sock.close()

        return scanners

    def _send_command(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a command over TCP and receive response.

        :param request: Request dictionary.
        :return: Response dictionary.
        """
        packed = msgpack.packb(request)
        header = struct.pack('>I', len(packed))

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.ip, self.commands_port))
            sock.sendall(header + packed)
            # Read header
            header_data = sock.recv(4)
            if len(header_data) != 4:
                raise ProfiTalkError("Incomplete header received")
            body_size = struct.unpack('>I', header_data)[0]
            # Read body
            body = b''
            while len(body) < body_size:
                chunk = sock.recv(body_size - len(body))
                if not chunk:
                    raise ProfiTalkError("Connection closed prematurely")
                body += chunk
            response = msgpack.unpackb(body, raw=False)
            if 'result' in response and response['result'] != 'RF_OK':
                raise ProfiTalkError(f"Error: {response['result']}")
            return response        

    def read_parameters_description(self) -> Dict[str, Any]:
        """Read parameters description."""
        request = {"request": "READ_PARAMETERS_DESCRIPTION"}
        return self._send_command(request)['payload']

    def read_parameters(self, names: List[str] = None, indexes: List[int] = None) -> Dict[str, Any]:
        """
        Read parameters values.

        :param names: List of parameter names.
        :param indexes: List of parameter indexes.
        :return: Dictionary of parameter values.
        """
        payload = {}
        if names:
            payload['names'] = names
        if indexes:
            payload['indexes'] = indexes
        request = {"request": "READ_PARAMETERS", "payload": payload}
        return self._send_command(request)['payload']
    
    def read_single_parameter(self, name: str = None, index: int = None):
        payload = {}
        if name:
            payload['name'] = name
        if index:
            payload['index'] = index
        request = {"request": "READ_PARAMETERS", "payload": payload}
        return self._send_command(request)['payload']
        pass

    def write_parameters(self, params: Dict[str, Any]) -> Dict[str, str]:
        """
        Write parameters.

        :param params: Dictionary of parameter name/value.
        :return: Dictionary of write results.
        """
        request = {"request": "WRITE_PARAMETERS", "payload": params}
        return self._send_command(request)['payload']

    def save_current_parameters(self):
        """Save current parameters to user area."""
        request = {"request": "SAVE_CURRENT_PARAMETERS"}
        self._send_command(request)

    def save_recovery_parameters(self):
        """Save current parameters to recovery area."""
        request = {"request": "SAVE_RECOVERY_PARAMETERS"}
        self._send_command(request)

    def load_recovery_parameters(self):
        """Load parameters from recovery area."""
        request = {"request": "LOAD_RECOVERY_PARAMETERS"}
        self._send_command(request)

    def reboot_device(self):
        """Reboot the device."""
        request = {"request": "REBOOT_DEVICE"}
        self._send_command(request)

    def read_profiles_dump(self) -> List[Dict[str, Any]]:
        """
        Read profiles dump.

        :return: List of profiles.
        """
        request = {"request": "READ_PROFILES_DUMP"}
        response = self._send_command(request)
        # Assuming it's returned as fragments, but manual shows per call a fragment?
        # Manual says command returns fragments, but in example it's one response.
        # Might need to loop until last=True
        profiles = []
        while True:
            payload = response['payload']
            for pr in payload['profiles']:
                print(f'index: {payload['index']}, len: {len(payload['profiles'])}')
                pass
            profiles.extend(payload['profiles'])
            if payload['last']:
                break
            # Send again? Manual doesn't specify if multiple calls needed.
            # Assuming single call returns all, or adjust if needed.
            # raise NotImplementedError("Multi-fragment dump not fully implemented")
        return profiles

    def _stream_reader(self, port: int, queue: queue.Queue):
        """Thread to read stream from TCP port."""        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.ip, port))
            while True:
                # Read header
                header_data = sock.recv(4)
                if len(header_data) != 4:                    
                    break
                body_size = struct.unpack('>I', header_data)[0]
                # Read body
                body = b''
                while len(body) < body_size:
                    chunk = sock.recv(body_size - len(body))
                    if not chunk:
                        break
                    body += chunk
                if len(body) != body_size:
                    break
                message = msgpack.unpackb(body, raw=False)
                queue.put(message)
    
    @staticmethod         
    def convert_dump_bin_to_np(data: bytes):
        profiles = []        

        index = 0
        while index < len(data):
            # Проверяем маркер начала
            if data[index] != 0x85:
                index += 1
                continue

            # Читаем количество точек (2 байта, little-endian)
            if index + 3 > len(data):
                break
            point_count = struct.unpack_from('<H', data, index + 1)[0]

            # Вычисляем размер профиля
            profile_size = 27 + point_count * 8  # 27 байт заголовка + N * (X+Z)
            if index + profile_size > len(data):
                print("Ошибка: недостаточно данных для полного профиля.")
                break

            # Извлекаем весь профиль
            profile_data = data[index : index + profile_size]

            # Парсим точки: начиная с байта 27
            points = []
            for i in range(point_count):
                x = struct.unpack('<f', profile_data[27 + i*8   : 27 + i*8 + 4])[0]
                z = struct.unpack('<f', profile_data[27 + i*8+4 : 27 + i*8 + 8])[0]
                points.append([x, z])

            profiles.append(np.array(points))
            index += profile_size  # переходим к следующему профилю

        return profiles
    
    @staticmethod
    def process_single_profile(prof: dict):
        points = []   
        
        if prof.get('intensity') is not None:
            use_intensity = True
            tmp_intensity = list(prof['intensity'])
            intensity = []
        else:
            use_intensity = False            
            intensity = None
        if prof['format']=='DATA_FORMAT_METRIC':
            ln = len(prof['profile'])
            for i in range(0,ln,4):
                x,z = struct.unpack('<hH', prof['profile'][i:i+4])  
                #tmp_y = prof['encoder_value']
                tmp_y = prof['measure_index']              
                points.append([x*prof['scaling'],tmp_y,z*prof['scaling']])
                if use_intensity:
                    intensity.append([x*prof['scaling'],tmp_y,tmp_intensity[i//4]])
            pass
        return points, intensity
    
    def enable_profile_dump(self):
        return self.write_parameters({'user_dump_enabled':True})
    
    def is_dump_ready(self):
        
        user_dump_enabled = self.read_single_parameter('user_dump_enabled')['user_dump_enabled']
        user_dump_capacity = self.read_single_parameter('user_dump_capacity')['user_dump_capacity']
        user_dump_size = self.read_single_parameter('user_dump_size')['user_dump_size']        
        
        # request parameters by list return error
        # self.read_parameters(names=[
        #     'user_dump_enabled',
        #     'user_dump_capacity',
        #     'user_dump_size',
        # ])
        
        if user_dump_enabled:
            return False
        if user_dump_size!=user_dump_capacity:
            return False
        return True

                
    def get_dump_http(self):
        url = f'http://{self.ip}/dump.bin'
        r = requests.get(url)
        data = self.convert_dump_bin_to_np(r.content)
        return data

    def get_profiles_stream(self, buffered: bool = True):
        """
        Get profiles stream as a generator.

        :param buffered: Use queue for buffering.
        :yield: Profile dictionaries.
        """
        q = queue.Queue()
        thread = threading.Thread(target=self._stream_reader, args=(self.profiles_port, q), daemon=True)
        thread.start()
        while True:
            yield q.get()
            
    def collect_profiles_stream(self, num_profiles: Optional[int] = None, buffered: bool = True):
        profiles = []
        q = multiprocessing.Queue()
        # thread = threading.Thread(target=self._stream_reader, args=(self.profiles_port, q), daemon=True)
        thread = multiprocessing.Process(
            target=self._stream_reader,
            args=(self.profiles_port, q),
            daemon=True
        )
        thread.start()
        count = 0        
        while True:
            profiles.append(q.get())
            count += 1
            print(count)
            if num_profiles is not None and count >= num_profiles:                
                break
            if num_profiles is None:
                break    
        thread.terminate()              
        return profiles        

    def get_video_stream(self, send_period_ms: Optional[int] = None, buffered: bool = True):
        """
        Get video frames stream as a generator.

        :param send_period_ms: Optional period for sending frames.
        :param buffered: Use queue for buffering.
        :yield: Video frame dictionaries.
        """
        q = queue.Queue()
        thread = threading.Thread(target=self._stream_reader, args=(self.video_port, q), daemon=True)
        thread.start()
        if send_period_ms is not None:
            # Send period command, but manual says for video stream, send inside the connection?
            # Assuming send over the same socket before reading.
            # But to set period, pack and send with header.
            packed = msgpack.packb({"send_period_ms": send_period_ms})
            header = struct.pack('>I', len(packed))
            # Need to send on the socket, but since thread started, maybe separate.
            # For simplicity, assume default or set via parameters.
            pass  # Implement if needed
        while True:
            yield q.get()

    def reset_network_parameters(self, serial: int):
        """Reset network parameters (via UDP)."""
        request = {"request": "RESET_NETWORK_PARAMETERS", "serial": serial}
        packed = msgpack.packb(request)

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        broadcast_addresses = ['255.255.255.255']
        for bcast in broadcast_addresses:
            sock.sendto(packed, (bcast, self.DEFAULT_SEARCH_PORT))
        sock.close()

# Example usage:
# scanners = ProfiTalkClient.discover_scanners()
# client = ProfiTalkClient(scanners[0]['ip4_addr'])
# params_desc = client.read_parameters_description()