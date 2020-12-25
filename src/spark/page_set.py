import logging

import pyspark.sql.functions as f

from .functions import shuffle_df, add_seq_col


def sub_list(a, b): return list(set(a) - set(b))


class PageSet:
    def __init__(self, data_frame, page_size, seq_col='row_seq', shuffle=False, page_count=None):
        self.seq_col = seq_col
        self.page_size = page_size

        # Shuffle when is specified...
        if shuffle:
            data_frame = shuffle_df(data_frame)

        # Exclude seq_col...
        data_frame = data_frame.select(sub_list(data_frame.columns, [seq_col]))

        # Add sequence column...
        self.data_frame = add_seq_col(data_frame, seq_col)

        total_count = self.data_frame.count()

        # Get pages count...
        if page_count:
            self.page_count = page_count
        else:
            self.page_count = int(total_count / page_size)

        self.__check_page_size(total_count, self.page_count, self.page_size)

        self._logger = logging.getLogger(f'page-set-{id(self)}')
        self._logger.debug(f'Page Size: {page_size}')
        self._logger.debug(f'Pages Count: {self.page_count}')
        self._logger.debug(f'Total elements: {self.data_frame.count()}')

    def __check_page_size(self, total_count, page_count, page_size):
        calculated_page_size = int(total_count / page_count)
        assert calculated_page_size == page_size, f'Unexpected page size {calculated_page_size} != {page_size}'

    def columns(self):
        return sub_list(self.data_frame.columns, [self.seq_col])

    def size(self):
        return self.page_count

    def shuffled(self):
        return PageSet(self.data_frame, self.page_size, self.seq_col, True, self.page_count)

    def get(self, number, seq_col=False):
        start = number * self.page_size
        end = start + (self.page_size - 1 if self.page_size > 1 else 0)
        
        page = self.data_frame.where(f.col(self.seq_col).between(start, end))

        self._logger.debug('Get page number: %s from %s to %s with %s size', number, start, end, page.count())

        return page if seq_col else page.select(self.columns())

    def __delitem__(self, key):
        raise Exception('Can not modify an immutable PageSet instance!')

    def __getitem__(self, number):
        return self.get(number)

    def __setitem__(self, key, value):
        raise Exception('Can not modify an immutable PageSet instance!')
