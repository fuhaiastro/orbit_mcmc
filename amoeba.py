import numpy as np

def amotry_sa(p, y, psum, func, ihi, fac, temperature, p_u, p_l, pars, chi2):
    """
    Helper function for amoeba_sa. Extrapolates by a factor fac through the face of the simplex
    across from the high point, tries it, and replaces the high point if the new point is better.
    
    Parameters:
    -----------
    p : ndarray
        Simplex vertices, shape (ndim, ndim+1).
    y : ndarray
        Function values at simplex vertices, shape (ndim+1,).
    psum : ndarray
        Sum of simplex vertices along second dimension, shape (ndim,).
    func : callable
        Function to minimize, takes an ndim vector and returns a scalar.
    ihi : int
        Index of the highest (worst) function value.
    fac : float
        Extrapolation factor.
    temperature : float
        Simulated annealing temperature.
    p_u, p_l : ndarray
        Upper and lower bounds for parameters, shape (ndim,).
    pars : list
        List to store parameter sets.
    chi2 : list
        List to store chi-squared values.
    
    Returns:
    --------
    float
        Function value at the trial point.
    """
    ndim = len(psum)
    fac1 = (1.0 - fac) / ndim
    fac2 = fac1 - fac
    # Compute trial point, respecting bounds
    ptry = np.maximum(np.minimum(psum * fac1 - p[:, ihi] * fac2, p_u), p_l)
    ytry = func(ptry)  # Evaluate function at trial point
    
    # Save parameters and chi-squared
    pars.append(ptry)
    chi2.append(ytry)
    
    # Apply simulated annealing: subtract a random log-distributed number
    ytry += temperature * np.log(np.random.uniform(0, 1, 1))[0]
    
    if ytry < y[ihi]:
        # If better than highest, replace highest
        y[ihi] = ytry
        psum += ptry - p[:, ihi]
        p[:, ihi] = ptry
    
    return ytry

def amoeba_sa(ftol, function_name=None, temperature=1.0, p0=None, scale=None, simplex=None,
              upper=None, lower=None, nmax=5000):
    """
    Multidimensional minimization of a function using the downhill simplex method
    with simulated annealing.
    
    Parameters:
    -----------
    ftol : float
        Fractional tolerance for convergence.
    function_name : callable, optional
        Function to minimize, takes an ndim vector and returns a scalar. Default is 'func'.
    temperature : float, optional
        Simulated annealing temperature. Default is 1.0.
    p0 : ndarray, optional
        Initial starting point, shape (ndim,).
    scale : float or ndarray, optional
        Characteristic length scale for each dimension, scalar or shape (ndim,).
    simplex : ndarray, optional
        Initial simplex vertices, shape (ndim, ndim+1).
    upper, lower : ndarray, optional
        Upper and lower bounds for parameters, shape (ndim,).
    nmax : int, optional
        Maximum number of function evaluations. Default is 5000.
    
    Returns:
    --------
    tuple
        (best_parameters, result_dict) where best_parameters is the parameter set at the
        global minimum, and result_dict contains:
        - function_value : Function values at final simplex points.
        - ncalls : Number of function evaluations.
        - pars : Array of parameter sets evaluated.
        - chi2 : Array of chi-squared values.
        - simplex : Final simplex vertices.
    """
    # Default function
    if function_name is None:
        def func(x): return x  # Placeholder; user must provide a function
    else:
        func = function_name
    
    # Initialize simplex
    if scale is not None and p0 is not None:
        ndim = len(p0)
        p = np.tile(p0, (ndim + 1, 1)).T  # Shape (ndim, ndim+1)
        scale = np.array(scale, dtype=np.float64)
        if scale.size == 1:
            scale = np.repeat(scale, ndim)
        for i in range(ndim):
            p[i, i + 1] = p0[i] + scale[i]
    elif simplex is not None:
        p = np.array(simplex, dtype=np.float64)
    else:
        raise ValueError("Either (scale, p0) or simplex must be initialized")
    
    # Validate simplex shape
    if p.ndim != 2:
        raise ValueError("Simplex must be a 2D array")
    ndim, mpts = p.shape
    if mpts != ndim + 1:
        raise ValueError("Simplex must have ndim+1 points")
    
    # Initialize bounds
    if upper is None:
        upper = np.full(ndim, np.inf)
    if lower is None:
        lower = np.full(ndim, -np.inf)
    upper = np.array(upper, dtype=np.float64)
    lower = np.array(lower, dtype=np.float64)

    # Clip within bounds
    for i in range(ndim):
        p[i,:] = np.clip(p[i,:], lower[i], upper[i]) 
    
    # Initialize function values
    y = np.array([func(p[:, i]) for i in range(mpts)], dtype=np.float64)
    pars = [p[:, i].copy() for i in range(mpts)]  # List of parameter sets
    chi2 = y.tolist()  # List of chi-squared values
    
    ncalls = 0
    psum = np.sum(p, axis=1)  # Sum along second dimension
    
    while ncalls <= nmax:
        # Apply simulated annealing to function values
        y -= temperature * np.log(np.random.uniform(0, 1, len(y)))
        
        # Find indices of lowest, highest, and next-highest function values
        s = np.argsort(y)
        ilo = s[0]
        ihi = s[ndim]
        inhi = s[ndim - 1]
        
        # Compute fractional tolerance
        d = abs(y[ihi]) + abs(y[ilo])
        rtol = 2.0 * abs(y[ihi] - y[ilo]) / d if d != 0.0 else ftol / 2.0
        
        # Check for convergence or max iterations
        if rtol < ftol or ncalls == nmax:
            idx = np.argmin(chi2)
            result = {
                'function_value': y,
                'ncalls': ncalls,
                'pars': np.array(pars).T,
                'chi2': np.array(chi2),
                'simplex': p
            }
            return pars[idx], result
        
        ncalls += 2
        # Try a reflection
        ytry = amotry_sa(p, y, psum, func, ihi, -1.0, temperature, upper, lower, pars, chi2)
        
        if ytry <= y[ilo]:
            # If better than best, try an expansion
            ytry = amotry_sa(p, y, psum, func, ihi, 2.0, temperature, upper, lower, pars, chi2)
        elif ytry >= y[inhi]:
            # If worse than next-highest, try a contraction
            ysave = y[ihi]
            ytry = amotry_sa(p, y, psum, func, ihi, 0.5, temperature, upper, lower, pars, chi2)
            if ytry >= ysave:
                # If still bad, contract around the best point
                for i in range(ndim + 1):
                    if i != ilo:
                        psum = 0.5 * (p[:, i] + p[:, ilo])
                        p[:, i] = psum
                        y[i] = func(psum)
                        pars.append(psum)
                        chi2.append(y[i])
                ncalls += ndim
                psum = np.sum(p, axis=1)
        else:
            ncalls -= 1  # No additional evaluation needed
    
    # Return the parameters corresponding to the global minimum chi-squared
    idx = np.argmin(chi2)
    result = {
        'function_value': y,
        'ncalls': ncalls,
        'pars': np.array(pars).T,
        'chi2': np.array(chi2),
        'simplex': p
    }
    return pars[idx], result