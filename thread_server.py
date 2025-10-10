import socket
from threading import Thread
from time import sleep


class Server(Thread):
    def __init__(self, ip, port):
        super().__init__()

        address = (ip, port)
        self.socket = socket.socket()  # 创建socket对象
        self.recv_thread_dict = {}
        self.socket.bind(address)  # 绑定端口
        self.client_socket_list = []
        self.socket.listen(5)  # 等待客户端连接，参数为TCP连接队列的大小,就是连接数

        self.sending_flag = False
        self.data = ""
        self.speaking_flag = False
        self.done_index = 0
        self.condition = 0
        self.background_label = ""
        self.background_index = 0
        print('Sever is listening...')

    def run(self):
        Thread(target=self.handle_send_client).start()
        while True:
            self.get_client_socket()
            sleep(0.1)

    def handle_send_client(self):
        while True:
            for client_socket in self.client_socket_list:
                try:
                    if self.sending_flag:
                        client_socket.send(self.data.encode('utf-8'))
                except Exception as e:
                    print(e)
                    client_socket.close()
                    self.client_socket_list.remove(client_socket)
            self.sending_flag = False
            sleep(0.1)

    def handle_receive_client(self, client_socket):
        while True:
            try:
                data = client_socket.recv(1024)
            except Exception as e:
                print(e)
                del self.recv_thread_dict[id(client_socket)]
                self.client_socket_list.remove(client_socket)
                return
            if len(data):
                data_str = data.decode("utf-8")
                print(f"Received {data_str}!")
                if "Speaking Flag" in data_str:
                    self.speaking_flag = True
                if "Cur index" in data_str:
                    self.done_index = int(data_str.split(":")[-1])
                if "Condition" in data_str:
                    self.condition = int(data_str.split(":")[-1])
                if "Background" in data_str:
                    data_str = data_str.split(":")[-1].split(" ")
                    self.background_label = data_str[0].lower()
                    self.background_index = int(data_str[1])
            else:
                del self.recv_thread_dict[id(client_socket)]
                self.done_index = 0
                self.condition = 0
                self.client_socket_list.remove(client_socket)  # 已经断开
                return

    def get_client_socket(self):
        client_socket, client_address = self.socket.accept()  # 建立客户端链接
        self.done_index = 0
        self.recv_thread_dict[id(client_socket)] = Thread(target=self.handle_receive_client, args=(client_socket,))
        self.recv_thread_dict[id(client_socket)].start()
        self.client_socket_list.append(client_socket)
        print(f'Client addr:{client_address}')

    def send_data(self, data: str = ""):
        self.sending_flag = True
        self.data = data

    def close(self):
        for client_socket in self.client_socket_list:
            client_socket.close()
        self.client_socket_list.clear()


if __name__ == '__main__':
    server = Server("127.0.0.1", 8012)
    server.start()
    while True:
        print(server.speaking_flag)
        sleep(0.1)
        # sleep(0.01)
        # if keyboard.is_pressed("q"):
        #     server.send_data(f"Audio length:{5.14:.4f}")
        #     sleep(1)
        #     server.send_data(f"Gesture index:{1}")
        #     sleep(1)
        #     server.send_data(f"Answer length:{7}")  # 发送总长度
        #     sleep(1)
        #     server.send_data(f"Async flag:{1}")
        #     sleep(1)
