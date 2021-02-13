import logging
import humanize

import pyspark.sql.functions as f

from util.profiler import Profiler
from .functions import shuffle_df, add_seq_col


def sub_list(a, b): return list(set(a) - set(b))


class NotFoundPageException(Exception):
    def __init__(self, page_number, pages_count):
        self.page_number = page_number
        self.pages_count = pages_count
        self.message = f'Not found {page_number} page! Pages: (0, {pages_count - 1})'
        super().__init__(self.message)


class PageSet:
    def __init__(self, data_frame, page_size, row_seq='row_seq', shuffle=False, page_count=None):
        self.__row_seq = row_seq
        self.__page_size = page_size

        # Shuffle when is specified...
        if shuffle:
            data_frame = shuffle_df(data_frame)

        # Exclude seq_col...
        data_frame = data_frame.select(sub_list(data_frame.columns, [row_seq]))

        # Add sequence column...
        self.__data_frame = add_seq_col(data_frame, row_seq).cache()

        self.__elements_count = self.__data_frame.count()

        # Get pages count...
        if page_count:
            self.__pages_count = page_count
        else:
            pages_count = self.__elements_count / self.__page_size
            self.__pages_count = int(pages_count) if pages_count % 2 == 0 else int(pages_count) + 1

        self._logger = logging.getLogger(f'page-set-{id(self)}')
        self._logger.info(f'Page Size: {humanize.intword(self.__page_size)}')
        self._logger.info(f'Pages Count: {humanize.intword(self.__pages_count)}')
        self._logger.info(f'Total elements: {humanize.intword(self.__elements_count)}')

    def columns(self):
        return sub_list(self.__data_frame.columns, [self.__row_seq])

    def size(self):
        return self.__pages_count

    def elements_count(self):
        return self.__elements_count

    def shuffled(self):
        return PageSet(self.__data_frame, self.__page_size, self.__row_seq, True, self.__pages_count)

    def get(self, number, seq_col=False):
        start = number * self.__page_size
        end = start + (self.__page_size - 1 if self.__page_size > 1 else 0)

        # with Profiler('Get page'):
        page = self.__data_frame.where(f.col(self.__row_seq).between(start, end))

        if len(page.head(1)) == 0:
            raise NotFoundPageException(number, self.__pages_count)

        self._logger.debug(
            'Get page number: %s from %s to %s with %s size. Time: %s ms',
            number,
            start,
            end,
            page.count()
        )

        result = page if seq_col else page.select(self.columns())

        return result

    def __delitem__(self, key):
        raise Exception('Can not modify an immutable PageSet instance!')

    def __getitem__(self, number):
        return self.get(number)

    def __setitem__(self, key, value):
        raise Exception('Can not modify an immutable PageSet instance!')
