
import numpy as np
from oitg.fitting.lorentzian import lorentzian

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    p = {}
    p['x0'] = 3
    p['y0'] = 1
    p['fwhm'] = 2.3
    p['a'] = 2
    
    
    n = 1001
    x = np.linspace(p['x0']-5, p['x0']+10, n)
    y = lorentzian.fitting_function(x, p) + 0.5*np.random.normal(size=n)
    

    p, p_error, x_fit, y_fit = lorentzian.fit(x, y, evaluate_function=True)
    
    print(p)
    print(p_error)
    
    plt.plot(x, y, '.')
    plt.plot(x_fit, y_fit)
    plt.show()