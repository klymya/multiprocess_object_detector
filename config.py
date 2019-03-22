import configparser
import os

CONFIG_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                           'config.ini')

config = configparser.ConfigParser()
config.read(CONFIG_PATH)
