
import numpy as np
from oitg.fitting.cos_2 import cos_2_fft as cos_2

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    
    n = 201
    x = np.linspace(0, 2, n)
    y = (np.cos(x*6. + 2)**2) + 0.1*np.random.normal(size=n) + 100.
    
    y_err = 0.1*np.random.normal(size=n)
    
    #p, p_error, x_fit, y_fit = cos.fit(x, y, evaluate_function=True,
    #        initialise={'period':5})
    p, p_error, x_fit, y_fit = cos_2.fit(x, y, evaluate_function=True)
    
    print(p)
    print(p_error)
    
    plt.errorbar(x, y, yerr=y_err, fmt='.')
    plt.plot(x_fit, y_fit)
    
    # Weighted fit
    p, p_error, x_fit, y_fit = cos_2.fit(x, y, y_err=y_err,
        evaluate_function=True)
    
    print(p)
    print(p_error)
    plt.plot(x_fit, y_fit)

    plt.show()