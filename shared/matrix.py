from .vector import Vector


class Matrix:
    vectors: list[Vector]

    def __init__(self, vectors: list[Vector] or list[list[int or float]] = None):
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

    def shape(self):
        return (len(self.vectors), len(self.vectors[0]))

    def __mul__(self, value: "Matrix" or Vector or list[float] or int or float):
        """
        Multiply matrix to value.

        Could be another matrix, vector or scalar.
        """
        # Check if the value is another matrix
        if isinstance(value, Matrix):
            new_matrix_vectors: list[Vector] = []
            transposed_matrix = value.transpose()

            for row_index in range(len(self.vectors)):
                new_row: list[int or float] = []

                for column_index in range(len(value.vectors[0])):
                    new_row.append(
                        self.vectors[row_index]
                        * transposed_matrix.vectors[column_index]
                    )

                new_matrix_vectors.append(Vector(new_row))

            return Matrix(new_matrix_vectors)

        # Otherwise we have a vector or array of numbers or scalar

        # Check if it's an array of numbers, then transform them into Vector
        if isinstance(value, list):
            value = Vector(value)

        # Check if the value is vector, then multiply by the vector
        if isinstance(value, Vector):
            return Vector([self.vectors[i] * value for i in range(len(self.vectors))])

        # Otherwise we have scalar
        return Matrix([vector * value for vector in self.vectors])

    def __sub__(self, other: "Matrix") -> "Matrix":
        return Matrix(
            [self.vectors[i] - other.vectors[i] for i in range(len(self.vectors))]
        )

    def __str__(self):
        return (
            "[\n" + ("\n".join(["  " + str(vector) for vector in self.vectors])) + "\n]"
        )

    def __repr__(self):
        return self.__str__()
