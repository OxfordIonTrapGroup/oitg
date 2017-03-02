
import numpy as np
from oitg.fitting.exponential_decay import exponential_decay

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    p = {}
    p['x0'] = 3
    p['y0'] = 1
    p['y_inf'] = 0
    p['tau'] = 2

    n = 1001
    x = np.linspace(p['x0']-5, p['x0']+20, n)
    y = exponential_decay.fitting_function(x, p)\
        + 0.05*np.random.normal(size=n)

    p, p_error, x_fit, y_fit = exponential_decay.fit(
        x, y, evaluate_function=True)

    print(p)
    print(p_error)

    plt.plot(x, y, '.')
    plt.plot(x_fit, y_fit)
    plt.show()
