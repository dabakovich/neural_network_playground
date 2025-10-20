class Vector:
    values: list[int or float]

    def __init__(self, values):
        self.values = values

    def multiply(self, other: "Vector") -> "Vector":
        if len(self.values) != len(other.values):
            raise ValueError("Vectors must have the same length")
        return Vector(
            [self.values[i] * other.values[i] for i in range(len(self.values))]
        )

    def clone(self) -> "Vector":
        return Vector([value for value in self.values])

    def __add__(self, other) -> "Vector":
        return Vector(
            [self.values[i] + other.values[i] for i in range(len(self.values))]
        )

    def __sub__(self, other) -> "Vector":
        return Vector(
            [self.values[i] - other.values[i] for i in range(len(self.values))]
        )

    def __mul__(self, other):
        if isinstance(other, Vector):
            if len(self.values) != len(other.values):
                raise ValueError("Vectors must have the same length")
            return sum(
                [self.values[i] * other.values[i] for i in range(len(self.values))]
            )

        return Vector([self.values[i] * other for i in range(len(self.values))])

    def __str__(self):
        return str([round(x, 2) for x in self.values])

    def __repr__(self):
        return self.__str__()

    def __len__(self) -> int:
        return len(self.values)
