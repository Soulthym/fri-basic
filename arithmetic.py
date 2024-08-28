from random import randrange

class Field:
    _gen = 5
    mod = 3 * 2 ** 30 + 1
    value: int

    def __new__(cls, value=None):
        if isinstance(value, Field):
            return value
        if value is None:
            value = cls.zero().value
        if not isinstance(value, int):
            raise TypeError(f"Can't convert {type(value)!r} to Field")
        self = super().__new__(cls)
        self.value = value % cls.mod
        return self

    @classmethod
    def gen(cls):
        return Field(cls._gen)

    def __repr__(self) -> str:
        return "%s %% %s" % (repr(self.value), self.mod)

    @staticmethod
    def zero():
        return Field(0)

    @staticmethod
    def one():
        return Field(1)

    @staticmethod
    def rand():
        return Field(randrange(0, Field.mod))

    def __neg__(self):
        return self.zero() - self
    def __add__(self, other):
        return Field(self.value + other)
    __radd__ = __add__

    def __sub__(self, other):
        return Field(self.value - other)

    def __rsub__(self, other):
        return -(self - other)

    def __mul__(self, other):
        return Field(self.value * other)
    __rmul__ = __mul__

    def __pow__(self, n):
        assert n >= 0
        cur_pow = self
        res = Field(1)
        while n > 0:
            if n % 2 != 0:
                res *= cur_pow
            n = n // 2
            cur_pow *= cur_pow
        return res

    def inverse(self):
        t, new_t = 0, 1
        r, new_r = Field.mod, self.value
        while new_r != 0:
            quotient = r / new_r
            t, new_t = new_t, (t - (quotient * new_t))
            r, new_r = new_r, r - quotient * new_r
        assert r == 1
        return Field(t)

    def __div__(self, other):
        return self * Field(other.inverse())

    def __eq__(self, value: object, /) -> bool:
        return self.value == Field(value).value

    def is_order(self, n):
        """
        Naively checks that the element is of order n by raising it to all powers up to n, checking
        that the element to the n-th power is the unit, but not so for any k<n.
        """
        assert n >= 1
        h = Field(1)
        for i in range(1, n):
            h *= self
            print(f"{i:<9}: {h}")
            if h == Field(1):
                return False
        return h * self == Field(1)

if __name__ == "__main__":
    g = Field.gen() ** (3*2**20)
    assert g.is_order(1024)
    print(f"{g=}")
    value = Field(1)
    print(f"{value=}")
    n = 0
    while True:
        n = n + 1
        value *= g
        print(f"{g}^{n}, {value=}")
        if n > 1 and value == g:
            break
    print("Cycle found after", n, "steps")
