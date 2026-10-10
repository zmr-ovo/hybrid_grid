"""Small integer arithmetic coder used by the Grid codec.

The coder is deliberately independent of PyTorch.  It consumes integer CDFs
whose first value is zero and whose last value is the common total frequency.
"""

from bisect import bisect_right


_STATE_BITS = 32
_FULL_RANGE = 1 << _STATE_BITS
_HALF_RANGE = _FULL_RANGE >> 1
_QUARTER_RANGE = _HALF_RANGE >> 1
_THREE_QUARTER_RANGE = _QUARTER_RANGE * 3
_MASK = _FULL_RANGE - 1


def probability_to_cdf(probabilities, precision=16):
    """Convert positive probabilities to a strictly increasing integer CDF."""
    if not isinstance(precision, int) or isinstance(precision, bool):
        raise TypeError("precision must be an integer")
    if precision < 1 or precision > 24:
        raise ValueError("precision must be between 1 and 24")

    values = [float(value) for value in probabilities]
    if not values:
        raise ValueError("probabilities must not be empty")
    if any(value < 0 or value != value or value == float('inf') for value in values):
        raise ValueError("probabilities must be finite and non-negative")
    total_probability = sum(values)
    if total_probability <= 0:
        raise ValueError("probabilities must contain positive mass")

    total = 1 << precision
    if len(values) > total:
        raise ValueError("precision is too small for the alphabet")

    # Reserve one count for every symbol, then distribute the remaining mass.
    remaining = total - len(values)
    scaled = [value / total_probability * remaining for value in values]
    counts = [1 + int(value) for value in scaled]
    missing = total - sum(counts)
    order = sorted(
        range(len(values)),
        key=lambda index: scaled[index] - int(scaled[index]),
        reverse=True,
    )
    for index in order[:missing]:
        counts[index] += 1

    cdf = [0]
    for count in counts:
        cdf.append(cdf[-1] + count)
    return tuple(cdf)


def _validate_cdf(cdf):
    values = tuple(int(value) for value in cdf)
    if len(values) < 2 or values[0] != 0:
        raise ValueError("cdf must start at zero and contain at least one symbol")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("cdf must be strictly increasing")
    if values[-1] > _QUARTER_RANGE:
        raise ValueError("cdf total is too large for the arithmetic coder")
    return values


class _BitWriter:
    def __init__(self):
        self.data = bytearray()
        self.current = 0
        self.count = 0

    def write(self, bit):
        self.current = (self.current << 1) | int(bit)
        self.count += 1
        if self.count == 8:
            self.data.append(self.current)
            self.current = 0
            self.count = 0

    def finish(self):
        if self.count:
            self.data.append(self.current << (8 - self.count))
        return bytes(self.data)


class _BitReader:
    def __init__(self, data):
        self.data = bytes(data)
        self.index = 0

    def read(self):
        byte_index, bit_index = divmod(self.index, 8)
        self.index += 1
        if byte_index >= len(self.data):
            return 0
        return (self.data[byte_index] >> (7 - bit_index)) & 1


class ArithmeticEncoder:
    """Encode a sequence one symbol at a time with caller-provided CDFs."""

    def __init__(self):
        self.low = 0
        self.high = _MASK
        self.pending = 0
        self.writer = _BitWriter()
        self.finished = False

    def _write_with_pending(self, bit):
        self.writer.write(bit)
        for _ in range(self.pending):
            self.writer.write(1 - bit)
        self.pending = 0

    def encode(self, symbol, cdf):
        if self.finished:
            raise RuntimeError("encoder has already been finished")
        cdf = _validate_cdf(cdf)
        if not isinstance(symbol, int) or isinstance(symbol, bool):
            raise TypeError("symbol must be an integer index")
        if symbol < 0 or symbol >= len(cdf) - 1:
            raise ValueError("symbol is outside the CDF alphabet")

        interval = self.high - self.low + 1
        total = cdf[-1]
        old_low = self.low
        self.low = old_low + interval * cdf[symbol] // total
        self.high = old_low + interval * cdf[symbol + 1] // total - 1

        while True:
            if self.high < _HALF_RANGE:
                self._write_with_pending(0)
            elif self.low >= _HALF_RANGE:
                self._write_with_pending(1)
                self.low -= _HALF_RANGE
                self.high -= _HALF_RANGE
            elif self.low >= _QUARTER_RANGE and self.high < _THREE_QUARTER_RANGE:
                self.pending += 1
                self.low -= _QUARTER_RANGE
                self.high -= _QUARTER_RANGE
            else:
                break
            self.low = (self.low << 1) & _MASK
            self.high = ((self.high << 1) & _MASK) | 1

    def finish(self):
        if self.finished:
            raise RuntimeError("encoder has already been finished")
        self.pending += 1
        self._write_with_pending(0 if self.low < _QUARTER_RANGE else 1)
        self.finished = True
        return self.writer.finish()


class ArithmeticDecoder:
    """Decode symbols in the same order and with the same CDFs as the encoder."""

    def __init__(self, data):
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data must be bytes-like")
        self.reader = _BitReader(data)
        self.low = 0
        self.high = _MASK
        self.code = 0
        for _ in range(_STATE_BITS):
            self.code = (self.code << 1) | self.reader.read()

    def decode(self, cdf):
        cdf = _validate_cdf(cdf)
        interval = self.high - self.low + 1
        total = cdf[-1]
        value = ((self.code - self.low + 1) * total - 1) // interval
        symbol = bisect_right(cdf, value) - 1
        if symbol < 0 or symbol >= len(cdf) - 1:
            raise ValueError("invalid arithmetic-coded stream")

        old_low = self.low
        self.low = old_low + interval * cdf[symbol] // total
        self.high = old_low + interval * cdf[symbol + 1] // total - 1

        while True:
            if self.high < _HALF_RANGE:
                pass
            elif self.low >= _HALF_RANGE:
                self.low -= _HALF_RANGE
                self.high -= _HALF_RANGE
                self.code -= _HALF_RANGE
            elif self.low >= _QUARTER_RANGE and self.high < _THREE_QUARTER_RANGE:
                self.low -= _QUARTER_RANGE
                self.high -= _QUARTER_RANGE
                self.code -= _QUARTER_RANGE
            else:
                break
            self.low = (self.low << 1) & _MASK
            self.high = ((self.high << 1) & _MASK) | 1
            self.code = ((self.code << 1) & _MASK) | self.reader.read()
        return symbol
