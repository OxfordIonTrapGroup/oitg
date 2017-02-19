
import numpy as np
from oitg.fitting.half_lorentzian import half_lorentzian_left

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    p = {}
    p['x0'] = 3
    p['y0'] = 1
    p['fwhm'] = 8
    p['a'] = 2
    
    
    n = 1001
    x = np.linspace(p['x0']-10, p['x0']+5, n)
    y = half_lorentzian_left.fitting_function(x, p) + 0.2*np.random.normal(size=n)
    

    p, p_error, x_fit, y_fit = half_lorentzian_left.fit(x, y, evaluate_function=True)
    
    print(p)
    print(p_error)
    
    plt.plot(x, y, '.')
    plt.plot(x_fit, y_fit)
    plt.show()