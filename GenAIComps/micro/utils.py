
from socket import AF_INET, SOCK_STREAM, socket
from typing import (
    List,
    Union,
)


def _single_port_free(host: str, port: int) -> bool:
    with socket(AF_INET, SOCK_STREAM) as session:
        if session.connect_ex((host, port)) == 0:
            return False
        else:
            return True


def is_port_free(host: Union[str, List[str]], port: Union[int, List[int]]) -> bool:
    if isinstance(port, list):
        if isinstance(host, str):
            return all([_single_port_free(host, _p) for _p in port])
        else:
            return all([all([_single_port_free(_h, _p) for _p in port]) for _h in host])
    else:
        if isinstance(host, str):
            return _single_port_free(host, port)
        else:
            return all([_single_port_free(_h, port) for _h in host])

