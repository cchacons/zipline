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


class Block(object):

    def __init__(self, array, days):
        self.array = array
        self.days = days

    def get_slice(self, start, end):
        block_slice = self.days.slice_indexer(start, end)
        return self.array[block_slice]


class USEquityHistoryLoader(object):

    def __init__(self, daily_reader, adjustment_reader):
        self._daily_reader = daily_reader
        self._adjustments_reader = adjustment_reader

        self._daily_window_blocks = {}

    def _ensure_block(self, asset, start, end, field):
        try:
            block = self._daily_window_blocks[asset]
            if start >= block.days[0] and end <= block.days[-1]:
                return
        except KeyError:
            pass

        col = getattr(USEquityPricing, field)
        cal = self._daily_reader._calendar
        prefetch_end = cal[min(cal.searchsorted(end) + 40, len(cal))]
        array = self._daily_reader.load_raw_arrays(
            [col], start, prefetch_end, [asset])
        days = cal[cal.slice_indexer(start, prefetch_end)]
        block = Block(array, days)
        self._daily_window_blocks[asset] = block

    def _window(self, asset, start, end, field):
        self._ensure_block(asset, start, end, field)
        self._daily_window_blocks[asset].get_slice(start, end)

    def history(self, asset, start, end, field):
        return self._window(asset, start, end, field)
