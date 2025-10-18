from .vector import Vector


class Matrix:
    vectors: list[Vector]

    def __init__(self, vectors: list[Vector] or list[list[int | float]] = None):
        if vectors is None:
            vectors = []

        if len(vectors) > 0 and isinstance(vectors[0], list):
            vectors = [Vector(vector) for vector in vectors]

        self.vectors = vectors

    def transpose(self) -> "Matrix":
        return Matrix(
            [
                [self.vectors[j].values[i] for j in range(len(self.vectors))]
                for i in range(len(self.vectors[0].values))
            ]
        )

    def scalar_multiply(self, scalar: float) -> "Matrix":
        return Matrix([vector * scalar for vector in self.vectors])

    def __mul__(self, vector: Vector or list[float]) -> Vector:
        if isinstance(vector, list):
            vector = Vector(vector)

        return Vector([self.vectors[i] * vector for i in range(len(self.vectors))])

    def __sub__(self, other) -> "Matrix":
        return Matrix(
            [self.vectors[i] - other.vectors[i] for i in range(len(self.vectors))]
        )

    def __str__(self):
        return "\n".join([str(vector) for vector in self.vectors])
