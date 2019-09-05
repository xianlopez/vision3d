import unittest
import ToolsP2
import numpy as np

class TestToolsP2(unittest.TestCase):
    def test_1(self):
        self.case_1()
        self.case_2()
        self.case_3()

    def case_1(self):
        X = np.zeros(shape=(3, 1))
        X[2, 0] = 1
        with self.assertRaises(AssertionError):
            T = ToolsP2.Normalization(X)

    def case_2(self):
        X = np.ones(shape=(3, 4))
        X[0, 0] = np.sqrt(2) + 2.5
        X[1, 0] = 0 - 3
        X[0, 1] = -np.sqrt(2) + 2.5
        X[1, 1] = 0 - 3
        X[0, 2] = 1 + 2.5
        X[1, 2] = -1 - 3
        X[0, 3] = -1 + 2.5
        X[1, 3] = 1 - 3
        X[:2, :] = X[:2, :] * 3.2
        T = ToolsP2.Normalization(X)
        Xnorm = np.matmul(T, X)
        expected_points = [(np.sqrt(2), 0, 1),
                           (-np.sqrt(2), 0,  1),
                           (1, -1, 1),
                           (-1, 1, 1)]
        for i in range(4):
            diff = np.mean(np.abs(Xnorm[:, i] - expected_points[i]))
            self.assertTrue(diff < 1e-6)

    def case_3(self):
        npoints = 20
        # Sample random points:
        X = np.reshape(10 * np.random.random_sample(3 * npoints), newshape=(3, npoints))
        # Create a copy of the points with the last coordinate set to 1:
        Y = np.copy(X)
        divisor = np.tile(np.expand_dims(Y[2, :], axis=0), reps=[3, 1])
        Y = np.divide(Y, divisor)
        # Normalization on the original points:
        T1 = ToolsP2.Normalization(X)
        Xnorm = np.matmul(T1, X)
        Xnorm = np.divide(Xnorm, np.expand_dims(Xnorm[2, :], axis=0))
        # Normalization on the points with last coordinate being 1:
        T2 = ToolsP2.Normalization(Y, last_coords_is_1=True)
        Ynorm = np.matmul(T2, Y)
        Ynorm = np.divide(Ynorm, np.expand_dims(Ynorm[2, :], axis=0))
        for i in range(npoints):
            diff = np.mean(np.abs(Xnorm[:, i] - Ynorm[:, i]))
            self.assertTrue(diff < 1e-6)


if __name__ == '__main__':
    unittest.main()



