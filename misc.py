import numpy as np
from skyfield.timelib import julian_day

def n(c):
    return ord(c) - (48 if c.isdigit() else 55)

def epoch2jd(s):
    """ Convert MPC epoch_packed to Julian Date """
    year = 100 * n(s[0]) + int(s[1:3])
    month = n(s[3])
    day = n(s[4])
    return julian_day(year, month, day) - 0.5

def convert_mean_anomaly(M1, a, JD1, JD2):
    """
    Convert the mean anomaly of an asteroid from one Julian Date to another.
    
    Parameters:
    M1 (float): Mean anomaly at JD1 (in degrees).
    a (float): Semi-major axis of the orbit (in AU).
    JD1 (float): Initial Julian Date.
    JD2 (float): Target Julian Date.
    
    Returns:
    float: Mean anomaly at JD2 (in degrees, normalized to [0, 360)).
    """
    
    k = 0.01720209895 # Gaussian gravitational constant in AU^3/2 per day
    # Calculate mean motion in degrees per day 
    n = np.sqrt(k**2 / a**3) * 180.0/np.pi
    
    # Calculate time difference in days
    delta_t = JD2 - JD1
    
    # Calculate change in mean anomaly
    delta_M = n * delta_t
    
    # Calculate new mean anomaly
    M2 = M1 + delta_M
    
    # Normalize to [0, 360)
    M2 = M2 % 360.0
    if M2 < 0:
        M2 += 360.0
        
    return M2