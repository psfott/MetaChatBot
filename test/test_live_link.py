from time import sleep

import json
import socket

from server import Server

if __name__ == "__main__":

    udpsocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udpsocket.bind(("localhost", 54322))
    target = ("localhost", 54321)

    server = Server("127.0.0.1", 8012)
    server.start()

    with open("json/bvh_test.json", "r") as f:
        json_str = json.loads(f.read())

    idx = 0

    sleep(10)

    server.send_data(f"Answer length and flag:{1} {1}")  # 发送总长度
    sleep(0.1)
    server.send_data(f"Streaming:{1}")

    for i in range(10):
        for data in json_str["test"]:
            udpsocket.sendto(json.dumps(data).encode(), target)
            sleep(1 / 60)
            print(f"Sending frame {idx}!")
            idx += 1
        sleep(3)
