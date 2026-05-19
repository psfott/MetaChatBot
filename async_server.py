import asyncio
import re
from threading import Thread
from time import sleep


class Server(Thread):
    def __init__(self, ip, port):
        super().__init__()
        self.message = None
        self.is_sending = False

        self.is_setting_background = False
        self.internal_server = None
        self.background_index = None
        self.background_label = None
        self.playing_type = None

        self.condition = None
        self.done_index = None
        self.patient_released_index = None
        self.clients = []
        self.ip = ip
        self.port = port

    def run(self):
        asyncio.run(self.main())

    async def handle_client(self, reader, writer):
        addr = writer.get_extra_info('peername')
        print(f'Connection from {addr}')
        self.clients.append(writer)
        try:
            while True:
                data = await reader.read(1024)
                if not data:
                    break
                data_str = data.decode("utf-8")
                print(f'Received from {addr}: {data_str}')

                cur_index_matches = re.findall(r"Cur index\s*:\s*(\d+)", data_str)
                if cur_index_matches:
                    self.done_index = int(cur_index_matches[-1])
                patient_released_matches = re.findall(r"Patient released\s*:\s*(\d+)", data_str, re.IGNORECASE)
                if patient_released_matches:
                    self.patient_released_index = int(patient_released_matches[-1])
                    print(f"Parsed patient_released_index={self.patient_released_index}")
                if "Condition" in data_str:
                    self.condition = int(data_str.split(":")[-1])
                if "Background" in data_str:
                    data_str = data_str.split(":")[-1].split(" ")
                    self.background_label = data_str[0].lower()
                    self.background_index = int(data_str[1])
                    self.playing_type = data_str[-1]
                    self.is_setting_background = True

        except Exception as e:
            print(f'Error: {e}')
        finally:
            print(f'Connection closed from {addr}')
            self.clients.remove(writer)
            writer.close()
            await writer.wait_closed()

    def send_data(self, message):
        self.is_sending = True
        self.message = message

    async def input_sender(self):
        while True:
            if self.is_sending:
                for client in self.clients:
                    client.write(self.message.encode())
                    await client.drain()
            self.is_sending = False
            await asyncio.sleep(0.0)  # 避免阻塞过久

    async def main(self):
        self.internal_server = await asyncio.start_server(self.handle_client, self.ip, self.port)
        addr = self.internal_server.sockets[0].getsockname()
        print(f'Server listening on {addr}')

        async with self.internal_server:
            await asyncio.gather(
                self.internal_server.serve_forever(),
                self.input_sender()
            )

    def close(self):
        self.internal_server.sockets[0].close()


if __name__ == "__main__":
    server = Server('127.0.0.1', 8012)
    server.start()
    while True:
        server.send_data("nihao")
        sleep(1)
