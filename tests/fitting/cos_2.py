
import numpy as np
from oitg.fitting.cos_2 import cos_2_fft as cos_2

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    
    n = 1001
    x = np.linspace(0, 10, n)
    y = (np.cos(x*6. + 2)**2) + 0.1*np.random.normal(size=n) + 100.
    
    #p, p_error, x_fit, y_fit = cos.fit(x, y, evaluate_function=True,
    #        initialise={'period':5})
    p, p_error, x_fit, y_fit = cos_2.fit(x, y, evaluate_function=True)
    
    print(p)
    print(p_error)
    
    plt.plot(x, y, '.')
    plt.plot(x_fit, y_fit)
    plt.show()