import math

GAMMA = 0.5772156649015328606  # Euler–Mascheroni constant
C = 2 ** (1/7)  # 2^(1/7)


def _approximate_L_of_X(X):
  return 7 * math.log2(((C - 1) / 75) * X + C)


def _X_of_L(L):
  """ Computes X(L) based on the given equation. """
  return (L**2)/8 - (9/40)*L + 75 * (2**(L/7) - C) / (C - 1) - GAMMA

def solve_L(X, L_init=None, tol=1e-6, max_iter=20):
  """ Newton's method to solve for L given X. """
  if L_init is None:
    L_init = _approximate_L_of_X(X)  # Use the approximation as the starting point
    
  def dX_dL(L):
    """ Derivative of X(L) with respect to L. """
    return (L / 4) - (9/40) + 75 * (2**(L/7) * math.log(2) / (7 * (C - 1)))
    
  L = L_init
  for _ in range(max_iter):
    f = _X_of_L(L) - X
    df = dX_dL(L)
    if abs(df) < 1e-10:
      break
    L -= f / df
    if abs(f) < tol:
      break
  return L

def calculate_skill_level(current_xp):
    # Hard cap: if current XP is equal to or exceeds 13034431, return level 99.
    if current_xp >= 13034431:
        return 99
    elif current_xp <= 0:
      return 1
    level = solve_L(current_xp)
    
    return int(round(level))

# At this point, having a static table with a lookup is probably faster/better
def get_xp_for_level(level):
    """Calculates the total XP needed to reach a given level."""
    if level <= 1:
        return 0
    if level >= 99:
        return 13034431 # Max XP at level 99
    X = _X_of_L(level)
    # Assuming _X_of_L directly corresponds to XP, may need adjustment
    return int(round(X))
