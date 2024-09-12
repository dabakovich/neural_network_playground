from neural_network.vector import Vector


class Matrix:
    def __init__(self, vectors: list[Vector] or list[list[float]]):
        if isinstance(vectors[0], list):
            vectors = [Vector(vector) for vector in vectors]

        self.vectors = vectors

    def transpose(self):
        return Matrix(
            [[self.vectors[j].values[i] for j in range(len(self.vectors))] for i in range(len(self.vectors[0].values))]
        )

    def scalar_multiply(self, scalar: float):
        return Matrix([vector * scalar for vector in self.vectors])

    def __mul__(self, vector: Vector or list[float]):
        if isinstance(vector, list):
            vector = Vector(vector)

        return Vector([self.vectors[i] * vector for i in range(len(self.vectors))])

    def __sub__(self, other):
        return Matrix([self.vectors[i] - other.vectors[i] for i in range(len(self.vectors))])

    def __str__(self):
        return '\n'.join([str(vector) for vector in self.vectors])
