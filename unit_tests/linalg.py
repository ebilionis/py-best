"""Unit-tests for best.linalg.

Author:
    Ilias Bilionis

Date:
    7/30/2013

"""


if __name__ == '__main__':
    import fix_path


import unittest
import numpy as np
import best.linalg


class BestLinAlgTest(unittest.TestCase):

    def test_kron_prod(self):
        print '-------------------------------'
        print 'Testing best.linalg.kron_prod()'
        print '-------------------------------'
        A1 = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
        A2 = A1
        A = (A1, A2)
        x = np.random.randn(A1.shape[1] * A2.shape[1])
        y = best.linalg.kron_prod(A, x)
        z = np.dot(np.kron(A1, A2), x)
        print 'Compare best.linalg.kron_prod(A, x):'
        print y
        print 'with np.dot(np.kron(A1, A2), x):'
        print z

    def test_kron_solve(self):
        print '--------------------------------'
        print 'Testing best.linalg.kron_solve()'
        print '--------------------------------'
        A1 = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
        A2 = A1
        A = (A1, A2)
        y = np.random.randn(A1.shape[1] * A2.shape[1])
        x = best.linalg.kron_solve(A, y)
        z = np.linalg.solve(np.kron(A1, A2), y)
        print 'Compare best.linalg.kron_solve(A, y):'
        print x
        print 'with np.linalg.solve(np.kron(A1, A2), y):'
        print z

    def test_update_cholesky(self):
        print '-------------------------------------'
        print 'Testing best.linalg.update_cholesky()'
        print '-------------------------------------'
        A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
        A_new = np.array([[2, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1],
                      [0, 0, -1, 2]])
        L = np.linalg.cholesky(A)
        B = A_new[:3, 3:]
        C = A_new[3:, 3:]
        L_new = best.linalg.update_cholesky(L, B, C)
        print 'Compare best.linalg.update_cholesky(L, B, C):'
        print L_new
        print 'with np.linalg.cholesky(A_new):'
        print np.linalg.cholesky(A_new)
        A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
        A_new = np.array([[2, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1],
                      [0, 0, -1, 2]])
        L = np.linalg.cholesky(A)
        B = A_new[:3, 3:]
        C = A_new[3:, 3:]
        L_new = best.linalg.update_cholesky(L, B, C)
        L_new_real = np.linalg.cholesky(A_new)
        y = np.random.randn(3)
        x = np.linalg.solve(L, y)
        z = np.random.randn(1)
        x_new = best.linalg.update_cholesky_linear_system(x, L_new, z)
        print 'Compare best.linalg.update_cholesky_linear_system(x, L_new, z):'
        print x_new
        print 'with np.linalg.solve(L_real_new, np.hstack([x, z]):'
        print np.linalg.solve(L_new_real, np.hstack([y, z]))

    def test_update_cholesky_linear_system(self):
        print '---------------------------------------------------'
        print 'Testing best.linalg.update_cholesky_linear_system()'
        print '---------------------------------------------------'




if __name__ == '__main__':
    unittest.main()