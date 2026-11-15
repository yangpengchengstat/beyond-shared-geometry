"""Fast deterministic algebra checks; no experiment, model, or calibration run.

These finite fixtures guard row/column conventions and factors in the appendix.
They are numerical regression checks for proved identities, not empirical tests
of the language-model activations and not a substitute for mathematical proof.
Run directly with Python or through pytest. Requires only NumPy.
"""

from itertools import permutations, product
import unittest

import numpy as np


def centered_frame(n):
    """Whitened centered n-state coordinates, of rank n-1."""
    contrasts = np.eye(n)[:, : n - 1] - np.eye(n)[:, [-1]]
    basis, _ = np.linalg.qr(contrasts, mode="reduced")
    return np.sqrt(n) * basis


def permutation_actions(n, dihedral=False):
    signs = (1, -1) if dihedral else (1,)
    indices = [tuple((sign * np.arange(n) + k) % n)
               for sign in signs for k in range(n)]
    matrices = [np.eye(n)[list(index)] for index in indices]
    return indices, matrices


def induced_actions(v, permutation_matrices):
    return [v.T @ p @ v / len(v) for p in permutation_matrices]


def frob_sq(matrix):
    return float(np.sum(matrix * matrix))


class TheoryIdentityChecks(unittest.TestCase):
    def assert_close(self, a, b, atol=2e-11):
        np.testing.assert_allclose(a, b, rtol=2e-11, atol=atol)

    def test_common_design_ridge_and_opposite_multiplication(self):
        indices, actions = permutation_actions(4, dihedral=True)
        p_index = {key: i for i, key in enumerate(indices)}
        base = np.arange(60, dtype=float).reshape(12, 5)
        full = np.sin(base / 7) + np.cos(base / 13) + np.eye(12, 5)
        deficient = full.copy()
        deficient[:, -1] = deficient[:, 0] - 2 * deficient[:, 1]
        for z in (full, deficient):
            alpha = 0.03
            identity = np.eye(z.shape[1])
            left = np.linalg.solve(z.T @ z + alpha * identity, z.T)
            residual = np.eye(len(z)) - z @ left
            block_actions = [np.kron(np.eye(3), p) for p in actions]
            fitted = [left @ p @ z for p in block_actions]
            for i, p in enumerate(block_actions):
                self.assert_close(z @ fitted[i] - p @ z, -residual @ p @ z)
                for j, q in enumerate(block_actions):
                    # Applying g then h maps state i to h(g(i)).
                    composed = tuple(indices[j][x] for x in indices[i])
                    k = p_index[composed]
                    self.assert_close(p @ q, block_actions[k])
                    self.assert_close(
                        fitted[i] @ fitted[j] - fitted[k],
                        -left @ p @ residual @ q @ z,
                    )
            # Positive ridge residuals are generally not projectors.
            self.assertGreater(np.linalg.norm(residual @ residual - residual), 1e-5)

    def test_stiefel_risk_via_exact_finite_second_moments(self):
        # The signed permutation ensemble has the same second moments needed
        # by the Haar calculation. This is an exact finite-sum fixture, not
        # Monte Carlo validation or sampling of a scientific null.
        n, r, d = 4, 2, 3
        v = np.sqrt(2.0) * np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
        _, permutations_c4 = permutation_actions(4)
        actions = induced_actions(v, permutations_c4)
        frames = []
        for perm in permutations(range(d)):
            for signs in product((-1., 1.), repeat=d):
                orthogonal = np.eye(d)[list(perm)] @ np.diag(signs)
                frames.append(orthogonal[:, :r])
        predictor = np.array([[.3, .2, -.7], [.5, -.4, .1], [.8, .9, -.2]])
        self.assert_close(np.mean([a @ a.T for a in frames], axis=0), r / d * np.eye(d))
        for q, p in zip(actions, permutations_c4):
            chi = np.trace(q)
            self.assert_close(np.mean([a @ q @ a.T for a in frames], axis=0), chi / d * np.eye(d))
            losses = []
            for a in frames:
                x = v @ a.T
                local = a @ q @ a.T + np.eye(d) - a @ a.T
                self.assert_close(x @ x.T, v @ v.T)
                self.assert_close(local @ local.T, np.eye(d))
                self.assert_close(x @ local, p @ x)
                losses.append(frob_sq(x @ predictor - p @ x) / (n * r))
            expected = 1 - (chi / r) ** 2 + frob_sq(predictor - chi / r * np.eye(d)) / d
            self.assert_close(np.mean(losses), expected)

    def test_centered_permutation_character_constants(self):
        for n, dihedral, all_expected, nonidentity_expected in (
            (7, False, 5 / 6, 35 / 36),
            (4, True, 7 / 9, 8 / 9),
        ):
            v = centered_frame(n)
            _, permutations_group = permutation_actions(n, dihedral)
            actions = induced_actions(v, permutations_group)
            characters = np.array([np.trace(q) for q in actions])
            r = n - 1
            for p, q, chi in zip(permutations_group, actions, characters):
                self.assert_close(v @ q, p @ v)
                self.assert_close(q @ q.T, np.eye(r))
                self.assert_close(chi, np.trace(p) - 1)
            self.assert_close(1 - np.mean((characters / r) ** 2), all_expected)
            self.assert_close(1 - np.mean((characters[1:] / r) ** 2), nonidentity_expected)

    def test_commutant_risk_and_commuting_orientation_control(self):
        v = centered_frame(4)
        _, permutations_d4 = permutation_actions(4, dihedral=True)
        actions = induced_actions(v, permutations_d4)
        r = v.shape[1]

        def project(matrix):
            return np.mean([q @ matrix @ q.T for q in actions], axis=0)

        vector = np.array([1., 2., 3.])
        reflection = np.eye(r) - 2 * np.outer(vector, vector) / np.dot(vector, vector)
        orientations = [np.eye(r), reflection, -np.eye(r)]
        weights = [.2, .3, .5]
        local = [[o @ q @ o.T for q in actions] for o in orientations]
        means = [sum(w * row[g] for w, row in zip(weights, local)) for g in range(len(actions))]
        direct = sum(w * sum(frob_sq(row[g] - means[g]) for g in range(len(actions)))
                     for w, row in zip(weights, local)) / (r * len(actions))
        commutant = 0.
        for w, o in zip(weights, orientations):
            for wp, op in zip(weights, orientations):
                u = o.T @ op
                projected = project(u)
                self.assert_close(project(projected), projected)
                self.assert_close(np.mean([frob_sq(q @ u - u @ q) for q in actions]),
                                  2 * frob_sq(u - projected))
                for q in actions:
                    self.assert_close(projected @ q, q @ projected)
                commutant += w * wp * frob_sq(u - projected) / r
        self.assert_close(direct, commutant)
        self.assertGreater(direct, 0.01)
        self.assert_close(project(-np.eye(r)), -np.eye(r))
        for first, commuting in zip(local[0], local[2]):
            self.assert_close(first, commuting)

    def test_finite_context_sample_mean_learning_curve(self):
        v = centered_frame(4)
        _, permutations_d4 = permutation_actions(4, dihedral=True)
        actions = induced_actions(v, permutations_d4)
        r = v.shape[1]
        o = np.diag([-1., 1., 1.])
        locals_ = np.array([actions, [o @ q @ o.T for q in actions]])
        center = np.mean(locals_, axis=0)
        b = np.mean([frob_sq(a - center) / (r * len(actions)) for a in locals_])
        for m in (1, 2, 3):
            training_risks, independent_risks = [], []
            for sequence in product((0, 1), repeat=m):
                sample = locals_[list(sequence)]
                estimate = np.mean(sample, axis=0)
                training_risks.append(np.mean([frob_sq(a - estimate) for a in sample]) / (r * len(actions)))
                independent_risks.append(np.mean([frob_sq(a - estimate) for a in locals_]) / (r * len(actions)))
            self.assert_close(np.mean(training_risks), (1 - 1 / m) * b)
            self.assert_close(np.mean(independent_risks), (1 + 1 / m) * b)


def test_theory_appendix_identities_without_fixtures():
    """Entry point for the repository's zero-argument CPU test runner."""
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TheoryIdentityChecks)
    suite.debug()


if __name__ == "__main__":
    unittest.main(verbosity=2)
