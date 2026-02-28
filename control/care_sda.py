import numpy as np


def care_sda(A: np.ndarray, H: np.ndarray, G: np.ndarray,
             r: float = 2.4, tol: float = 1e-9,
             max_iter: int = 50, raise_on_fail: bool = True) -> np.ndarray:
    """
    Solve the continuous-time Algebraic Riccati Equation (CARE) via the
    Structure-Preserving Doubling Algorithm (SDA):

        A.T @ X + X @ A - X @ G @ X + H = 0.

    Reference:
        "A structure-preserving doubling algorithm for continuous-time
        algebraic Riccati equations"
    """
    
    def _right_solve(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute A @ inv(B) without forming inv(B)"""
        return np.linalg.solve(B.T, A.T).T

    I = np.eye(A.shape[0])

    # Precomputation
    Ar = A - r * I
    Ar_t = Ar.T
    Ar_inv_G = np.linalg.solve(Ar, G)
    A_HAG = Ar_t + H @ Ar_inv_G

    # Initialization
    A_old = I + 2.0 * r * np.linalg.solve(Ar + G @ np.linalg.solve(Ar_t, H), I)
    G_old = 2.0 * r * _right_solve(Ar_inv_G, A_HAG)
    H_old = 2.0 * r * np.linalg.solve(A_HAG, _right_solve(H, Ar))

    for _ in range(max_iter):
        # Precomputation
        A_old_t = A_old.T
        I_HG = I + H_old @ G_old
        I_GH = I + G_old @ H_old

        # Update
        A_new = A_old @ np.linalg.solve(I_GH, A_old)
        G_new = G_old + A_old @ G_old @ np.linalg.solve(I_HG, A_old_t)
        H_new = H_old + A_old_t @ np.linalg.solve(I_HG, H_old @ A_old)

        # Computate matrix norm for convergence check
        diff = np.linalg.norm(H_new - H_old, ord="fro")
        norm_H = np.linalg.norm(H_new, ord="fro")

        # Save for next iteration
        A_old = A_new
        G_old = G_new
        H_old = H_new

        # Convergence check
        if diff <= (tol * norm_H):
            # Return symmetrized X = H_new
            return (H_new + H_new.T) / 2

    if raise_on_fail:
        raise RuntimeError("SDA did not converge within max_iter.")

    return None
