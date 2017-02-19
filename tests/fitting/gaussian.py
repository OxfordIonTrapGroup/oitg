
import numpy as np
from oitg.fitting.gaussian import gaussian

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    p = {}
    p['x0'] = 3
    p['y0'] = 1
    p['sigma'] = 2.3
    p['a'] = 2
    
    
    n = 1001
    x = np.linspace(p['x0']-5, p['x0']+10, n)
    y = gaussian.fitting_function(x, p) + 0.5*np.random.normal(size=n)
    
    #p, p_error, x_fit, y_fit = cos.fit(x, y, evaluate_function=True,
    #        initialise={'period':5})
    p, p_error, x_fit, y_fit = gaussian.fit(x, y, evaluate_function=True)
    
    print(p)
    print(p_error)
    
    plt.plot(x, y, '.')
    plt.plot(x_fit, y_fit)
    plt.show()