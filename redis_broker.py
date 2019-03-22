import json
import logging
import sys
from abc import ABC, abstractmethod
from logging.handlers import TimedRotatingFileHandler
from multiprocessing import JoinableQueue, Process

import numpy as np
from multiprocessing_logging import install_mp_handler
from redis import Redis

FORMATTER = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LOG_FILE = "logs.log"
LEVEL = logging.DEBUG


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_file_handler():
    file_handler = TimedRotatingFileHandler(LOG_FILE, when='midnight')
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setFormatter(FORMATTER)
    return file_handler


def get_logger(logger_name):
    logger = logging.getLogger(logger_name)

    logger.setLevel(LEVEL)  # better to have too much log than not enough

    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler())

    # with this pattern, it's rarely necessary to propagate the error up to parent
    logger.propagate = False

    return logger


logger = get_logger(__name__)
install_mp_handler()


class BaseModelProcessor(ABC):
    @abstractmethod
    def process(self, **kwargs):
        """
        :param kwargs:
        :return: response to redis producer
        """
        pass


class RedisHandler:

    def __init__(self, redis_host, redis_port, input_topic, output_topic, processes_num, ModelProcessor):
        """
        :param redis_host: ‘host’ host (string) of Redis server.
        :param redis_port: ‘port’ port (integer) of Redis server.
        :param input_topic: a key in redis from which should to get new message
        :param output_topic: a key in redis in which should put new message
        :param ModelProcessor: type for message processing. Implementation of
        "BaseModelProcessor"
        """
        if not issubclass(ModelProcessor, BaseModelProcessor):
            raise ValueError('{} should be subclass of {}'
                             .format(ModelProcessor, BaseModelProcessor))

        self.input_topic = input_topic
        self.output_topic = output_topic
        self.redis_host = redis_host
        self.redis_port = redis_port

        self.redis_consumer = Redis(host=self.redis_host, port=self.redis_port)

        self.queue = JoinableQueue()
        self.processes = []
        for i in range(processes_num):
            self.processes.append(Process(target=self._worker,
                                          args=(ModelProcessor,)))
            self.processes[-1].daemon = True
            self.processes[-1].start()

    def _worker(self, ModelProcessor):
        detector = ModelProcessor()
        redis_producer = Redis(host=self.redis_host, port=self.redis_port)

        for message in iter(self.queue.get, None):
            message = json.loads(message.decode('UTF-8'))

            response = detector.process(message)
            redis_producer.rpush(self.output_topic, json.dumps(response, cls=NumpyEncoder).encode('UTF-8'))

            self.queue.task_done()
        self.queue.task_done()

    def main_loop(self):
        try:
            while True:
                topic, message = self.redis_consumer.blpop(keys=[self.input_topic])
                self.queue.put(message)
        except Exception as e:
            logger.error('Exception in RedisHandler. Exception message: {}'.format(e))
        finally:
            # finish everything
            self.queue.join()

            for _ in self.processes:
                self.queue.put(None)
            self.queue.join()

            for p in self.processes:
                p.join()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)
