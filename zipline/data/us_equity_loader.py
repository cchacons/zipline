# Copyright 2015 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from zipline.pipeline.data.equity_pricing import USEquityPricing
#from zipline.lib._float64window import AdjustedArrayWindow as Float64Window
#from zilpnie.lib._int64window import AdjustedArrayWindow as Int64Window

##

#        return self._iterator_type(
#            data, np.array
#            self._viewtype, dtype('float64')
#            self.adjustments, Float64Multiply(first_row=0, last_row=0, first_col=0, last_col=0)
#            offset, int 
#            window_length, windowlength
#        )

class Block(object):

    def __init__(self, array, adjustments, cal_start, cal_end):
        self.array = array
        self.adjustments = adjustments
        self.cal_start = cal_start
        self.cal_end = cal_end

    def get_slice(self, start_ix, end_ix):
        return self.array[
            start_ix - self.cal_start:end_ix - self.cal_start + 1]


class USEquityHistoryLoader(object):

    def __init__(self, daily_reader, adjustment_reader):
        self._daily_reader = daily_reader
        self._calendar = daily_reader._calendar
        self._adjustments_reader = adjustment_reader

        self._daily_window_blocks = {}

    def _ensure_block(self, asset, start, end, start_ix, end_ix, field):
        try:
            block = self._daily_window_blocks[asset]
            if start_ix >= block.cal_start and end_ix <= block.cal_end:
                return
        except KeyError:
            pass

        col = getattr(USEquityPricing, field)
        cal = self._calendar
        prefetch_end_ix = min(end_ix + 40, len(cal) - 1)
        prefetch_end = cal[prefetch_end_ix]
        array = self._daily_reader.load_raw_arrays(
            [col], start, prefetch_end, [asset])[0][:, 0]
        days = cal[start_ix:prefetch_end_ix]
        adjs = self._adjustments_reader.load_adjustments([col], days, [asset])
        block = Block(array, adjs, start_ix, prefetch_end_ix)
        self._daily_window_blocks[asset] = block

    def history(self, asset, start, end, field):
        start_ix = self._calendar.get_loc(start)
        end_ix = self._calendar.get_loc(end)
        self._ensure_block(asset, start, end, start_ix, end_ix, field)
        return self._daily_window_blocks[asset].get_slice(start_ix, end_ix)
