import datetime as dt
import logging
import timeit

import humanize


class Profiler:
    def __init__(self, desc=''):
        self.__logger = logging.getLogger('Profiler')
        self.__desc = desc

    def __stop_time(self):
        stop_time = timeit.default_timer()
        return stop_time - self._start_time

    def __enter__(self):
        self._start_time = timeit.default_timer()
        return self

    def info(self, value):
        self.__logger.info(value)

    def __exit__(self, type, value, traceback):
        delta = dt.timedelta(seconds=self.__stop_time())
        
        humanized_delta = humanize.precisedelta(delta, minimum_unit="microseconds", format="%0.0f")
        
        self.__logger.info(f'Desc: \'{self.__desc}\' - Time: {humanized_delta}')
