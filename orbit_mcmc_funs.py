import numpy as np
from skyfield.timelib import julian_day
from skyfield.data import mpc
from skyfield.api import load, N, W, wgs84
from skyfield.constants import GM_SUN_Pitjeva_2005_km3_s2 as GM_SUN

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

def angular_separation(ra1, dec1, ra2, dec2):
    """
    Compute the angular separation between two points on the celestial sphere.
    
    Parameters:
    ra1, dec1 : float
        Right Ascension and Declination of the first object (in degrees).
    ra2, dec2 : float
        Right Ascension and Declination of the second object (in degrees).
    
    Returns:
    float
        Angular separation in degrees.
    """
    # Convert degrees to radians
    ra1_rad = np.radians(ra1)
    dec1_rad = np.radians(dec1)
    ra2_rad = np.radians(ra2)
    dec2_rad = np.radians(dec2)
    
    # Haversine formula
    delta_ra = ra2_rad - ra1_rad
    delta_dec = dec2_rad - dec1_rad
    
    a = np.sin(delta_dec / 2)**2 + np.cos(dec1_rad) * np.cos(dec2_rad) * np.sin(delta_ra / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Convert back to degrees
    return np.degrees(c)

def predict_pos(theta,elms,jds):
    # a, e, i, Omega, w, MA = theta
    elms.semimajor_axis_au = theta[0]
    elms.eccentricity = theta[1]
    elms.inclination_degrees = theta[2]
    elms.longitude_of_ascending_node_degrees = theta[3]
    elms.argument_of_perihelion_degrees = theta[4]
    elms.mean_anomaly_degrees = theta[5]
    # recompute daily motion (not used by mpcorb_orbit)
    # Gaussian gravitational constant in AU^3/2 per day: K = 0.01720209895
    #elms.mean_daily_motion_degrees = np.sqrt(0.01720209895**2 / theta[0]**3) * 180.0/np.pi
    
    # Build a Timescale for time conversions
    ts = load.timescale()
    # Build elliptical orbit
    asteroid = sun + mpc.mpcorb_orbit(elms, ts, GM_SUN)
    # Timescale.tt_jd: Build a Time object from a Terrestrial Time Julian date.
    t = ts.tt_jd(jds)
    # Predict astrometric RA & Dec (matching those from JPL Horizon)
    ra_pred, dec_pred, dis_pred = earth.at(t).observe(asteroid).radec()
    return ra_pred, dec_pred, dis_pred

def chi2fun(theta,elms,jds,ras,decs,perr):
    ra_pred, dec_pred, _ = predict_pos(theta,elms,jds)
    chi = angular_separation(ras,decs,ra_pred._degrees,dec_pred._degrees)/perr
    return np.sum(chi**2)

def ln_like(theta, bounds, elms,jds,ras,decs,perr):
    # only calculate when pars are within bounds
    if np.all((theta > bounds[0, :]) & (theta < bounds[1, :])):
        return -0.5 * chi2fun(theta,elms,jds,ras,decs,perr)
    return -np.inf