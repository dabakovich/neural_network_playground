class Vector:
    def __init__(self, values):
        self.values = values

    def multiply(self, other):
        if len(self.values) != len(other.values):
            raise ValueError('Vectors must have the same length')
        return Vector([self.values[i] * other.values[i] for i in range(len(self.values))])

    def __add__(self, other):
        return Vector([self.values[i] + other.values[i] for i in range(len(self.values))])

    def __sub__(self, other):
        return Vector([self.values[i] - other.values[i] for i in range(len(self.values))])

    def __mul__(self, other):
        if isinstance(other, Vector):
            if len(self.values) != len(other.values):
                raise ValueError('Vectors must have the same length')
            return sum([self.values[i] * other.values[i] for i in range(len(self.values))])

        return Vector([self.values[i] * other for i in range(len(self.values))])

    def __str__(self):
        return str([round(x, 2) for x in self.values])
