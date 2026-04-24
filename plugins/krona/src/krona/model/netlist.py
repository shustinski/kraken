from enum import Enum, auto


class Transistor:

    def __init__(self, name, mosfet_type:MosfetType):
        self.__name = name
        self.__type = mosfet_type

        self.__gate_logic_level: bool|None = None

        self.__source_net:str = ''
        self.__drain_net:str = ''
        self.__gate_net:str = ''

    def set_connection(self, source_net, drain_net, gate_net):
        self.__source_net = source_net
        self.__drain_net = drain_net
        self.__gate_net = gate_net

    def set_logic_level(self, value:bool):
        self.__gate_logic_level = value

    @property
    def open(self):
        return self.__gate_logic_level ^ self.__type.value

    @property
    def connections(self):
        return self.__source_net, self.__drain_net, self.__gate_net

class Port:

    def __init__(self, name, direction: PortDirection):
        self.__name = name
        self.__direction = direction
        self.__net:str = ''

    def connect_to_net(self, net:str):
        self.__net = net

    @property
    def net(self):
        return self.__net

    @property
    def direction(self):
        return self.__direction

    @property
    def name(self):
        return self.__name

class Net:

    def __init__(self, name):
        self.__name = name
        self.__type: NetType|None = None
        self.connections:list = []

    def set_name(self, name:str):
        self.__name = name

    @property
    def name(self):
        return self.__name





    @staticmethod
    def append_connections_from_text(text:str):
        """
        Добавляет соединения к текущим из фрагмента edf файла
        :param text: Фрагмент edf файла, содержащий информацию о net
        :return:
        """
        net_joined = EdifParser.find_classes(text, '(joined ')

    def append_connections(self, connections):
        pass

class PortDirection(Enum):
    IN = auto()
    OUT = auto()
    INOUT = auto()

class MosfetType(Enum):
    P = True
    N = False

class NetType(Enum):
    pwr = auto()
    gnd = auto()
    log = auto()
