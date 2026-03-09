"""Functions for generating partitions of integers under linear constraints."""

from functools import lru_cache


def generate_partitions(
    ab_pairs: list[tuple[int, int]],
    target_p: int,
    target_q: int,
    m: int,
    n: int
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Return a list of all pairs (k_tuple, l_tuple) that satisfy:
        sum(k_i) == target_p
        sum(l_i) == target_q
        sum(a_i * (k_i + l_i)) == m
        sum(b_i * (k_i + l_i)) == n

    IMPORTANT: This was vibe-coded so it may contain bugs!

    Args:
        ab_pairs: List of (a_i, b_i) pairs defining the constraints
        target_p: Target sum for k_i
        target_q: Target sum for l_i
        m: Target weighted sum for a_i * (k_i + l_i)
        n: Target weighted sum for b_i * (k_i + l_i)

    Returns:
        A list of all (k_tuple, l_tuple) pairs that satisfy the constraints

    """
    N = len(ab_pairs)

    # recursion with memoization; returns a tuple of (k_tuple, l_tuple) pairs for given state
    @lru_cache(maxsize=None)
    def rec(idx: int, sum_p: int, sum_q: int, sum_a: int, sum_b: int):
        # If we've placed values for all ab_pairs, check final constraints
        if idx == N:
            if sum_p == target_p and sum_q == target_q and sum_a == m and sum_b == n:
                return (((), ()), )  # a single solution: empty tails
            return ()  # no solutions

        a, b = ab_pairs[idx]

        # remaining budget for p,q
        rem_p = target_p - sum_p
        rem_q = target_q - sum_q
        if rem_p < 0 or rem_q < 0:
            return ()

        # max_total = maximum possible (k + ell) at this position without exceeding m or n
        if a > 0:
            max_by_a = (m - sum_a) // a
        else:
            max_by_a = rem_p + rem_q  # effectively unbounded by 'a'
        if b > 0:
            max_by_b = (n - sum_b) // b
        else:
            max_by_b = rem_p + rem_q
        max_total = min(max_by_a, max_by_b, rem_p + rem_q)

        results = []

        # iterate total = k + ell from 0 .. max_total.
        # For each total, k ranges from max(0, total - rem_q) to min(total, rem_p)
        for total in range(max_total + 1):
            kmin = max(0, total - rem_q)
            kmax = min(total, rem_p)
            for k in range(kmin, kmax + 1):
                ell = total - k
                new_sum_a = sum_a + a * total
                new_sum_b = sum_b + b * total
                # primal pruning: if these exceed m,n then skip
                if new_sum_a > m or new_sum_b > n:
                    continue

                # recurse for the rest. rec returns a tuple of tail solutions (k_tail, l_tail)
                tails = rec(idx + 1, sum_p + k, sum_q + ell, new_sum_a, new_sum_b)
                if not tails:
                    continue

                # attach current (k,ell) in front of each tail
                for tail_k, tail_l in tails:
                    results.append(((k,) + tail_k, (ell,) + tail_l))

        return tuple(results)

    # rec(0, 0, 0, 0, 0) returns tuple of solutions; convert to list for convenience
    sols = rec(0, 0, 0, 0, 0)

    return [(k_tuple, l_tuple) for k_tuple, l_tuple in sols]
