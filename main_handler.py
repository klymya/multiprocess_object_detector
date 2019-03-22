from time import sleep

from detection_unit.redis_broker import RedisHandler, get_logger
from detection_model import ObjectDetector
from config import config

logger = get_logger(__name__)


REDIS_HOST = config.get('redis', 'host')
REDIS_PORT = int(config.get('redis', 'port'))
INPUT_TOPIC = config.get('redis', 'input_topic')
OUTPUT_TOPIC = config.get('redis', 'output_topic')
PROCESSES_NUM = int(config.get('redis', 'processes_num'))


def wait_port_is_open(host, port):
    import socket
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                return
        except socket.gaierror:
            pass
        sleep(1)


if __name__ == '__main__':
    logger.info('Detection unit started.')
    wait_port_is_open(REDIS_HOST, REDIS_PORT)
    redis_handler = RedisHandler(REDIS_HOST, REDIS_PORT, INPUT_TOPIC, OUTPUT_TOPIC,
                                 PROCESSES_NUM, ObjectDetector)
    redis_handler.main_loop()
