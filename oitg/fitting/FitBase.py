

import numpy as np
from scipy.optimize import curve_fit


class FitError(Exception):
    pass


class FitParameters:
    """An object used to pass parameters to the fit function"""
    def __init__(self, names=[], constant_parameters=[],
                 initialised_parameters=[]):
        """Initialises a parameter object.

         - names: a sequence of all parameter names used
         - constant_parameters: an optional dictionary of those parameters
            which are to be held constant, and the corresponding values
         - initialised_parameters: an optional dictionary of those
            parameters which are to be manually initialised before
            the fit rather than automatically
        """

        self.parameter_dict = {}
        self.constant_parameter_names = set(constant_parameters.keys())
        self.initialised_parameter_names = set(initialised_parameters.keys())

        # Check the contant and initialisation parameters for dodgy entries
        names = set(names)
        undefined_constants = self.constant_parameter_names - names
        if undefined_constants:
            raise FitError("Parameters specified as constant "
                           "do not exist: {}".format(undefined_constants))
        undefined_initialised = self.initialised_parameter_names - names
        if undefined_initialised:
            raise FitError("Initial values specified for parameters that "
                           "do not exist: {}".format(undefined_initialised))

        for name in names:
            if name in self.constant_parameter_names:
                # If the parameter should be constant, set its value
                self.parameter_dict[name] = constant_parameters[name]
            elif name in self.initialised_parameter_names:
                # If the parameter should be initialised
                self.parameter_dict[name] = initialised_parameters[name]
            else:
                self.parameter_dict[name] = 0

        # A list of names of those parameters which are to be variables
        # i.e. all of the remaining parameters
        self.variable_names = list(names - self.constant_parameter_names)

        # Internally set to true to stop initialisation parameters from
        # being overwritten during auto-initialisation
        self.initialisation_mode = False

    def __getitem__(self, name):
        """Return the current value of a parameter"""
        return self.parameter_dict[name]

    def __setitem__(self, name, value):
        """Set the value of a parameter, unless that parameter is constant
        or not to be initialised"""

        # If the parameter is a constant, don't set it
        if name in self.constant_parameter_names:
            return

        # If we are in init mode and the parameter shouldn't be
        # initialised
        if self.initialisation_mode and \
           name in self.initialised_parameter_names:
            return

        # Otherwise, set the value
        self.parameter_dict[name] = value

    def absorb_variable_values(self, values):
        """Set the value of all variables with the given sequence of values.
        The variables are indexed with the order of the names in
        variable_names"""

        for i in range(len(self.variable_names)):
            name = self.variable_names[i]
            self.parameter_dict[name] = values[i]


class FitBase:
    """An object associated with a fitting function."""

    def __init__(self, parameter_names,
                 fitting_function,
                 parameter_initialiser=None,
                 derived_parameter_function=None,
                 parameter_bounds={}):
        """Create an object for fitting a function.

        - parameter_names:
            A sequence of the names of all parameters for this function
        - fitting_function:
            Handle to function to be fitted.
            function should have calling sign f(x, P)
            where x is a numpy array of x values
            and P is a FitParameters object
        - parameter_initialiser:
            An optional function used to initialise the parameters
            function should have calling sign f(x, y, P)
            where x and y are the numpy arrays for the data to be fitted
            and P is the FitParameters object
            If no function is given, the all non-initialised parameters
            are set to 0
        - derived_parameter_function:
            An optional function used to generate certain derived
            parameters after fitting has taken place
            i.e. for a gaussian parameterised with sigma, one might
            want to calculate the FWHM
            Callsign: p_dict, p_error_dict = f(p_dict, p_error_dict)
        - parameter_bounds:
            Allowed ranges for the parameters, as a dictionary of
            (lower_limit, upper_limit) tuples indexed by parameter name. Use
            +/- np.inf to disable either of the bounds.
        """

        self.parameter_names = parameter_names
        self.fitting_function = fitting_function
        self.parameter_initialiser = parameter_initialiser
        self.derived_parameter_function = derived_parameter_function
        self.parameter_bounds = parameter_bounds

    def fit(self, x, y, y_err=None,
            x_limit=[-np.inf, np.inf], y_limit=[-np.inf, np.inf],
            constants={}, initialise={},
            calculate_residuals=False,
            evaluate_function=False,
            evaluate_x_limit=[None, None],
            evaluate_n=1000):
        """Perform a fit of this object's function to the given data.

        - x and y are the arrays of data to fit to.

        - y_err is an optional array of the standard error of the
            y input data. If present, these errors are used as
            weights in the fit.

        - x/y_limit are the min and max values in the input data
            that should be used to fit to.
            data-points outside of these ranges are not included
            in the fit. If any of the limits are , then no
            limit is set.
            i.e. x_limit = [5, np.inf] sets a lower x limit of 5
                and no upper limit (np.inf = +ve infinity)

        - constants is a dictionary of those parameters which should
            be held constant during the fitting, along with their
            values

        - initialise is a dictionary of those parameters which should
            be initialise to specific values instead of using the
            auto-initialisation function

        - calculate_residuals should be set to True if the user
            wishes the residuals to be calculated and returned

        - evaluate_function should be set to True if the user
            wishes to evaluate the function after fitting and
            return the x and y values. This is useful for
            plotting fit functions.

        - evaluate_x_limit  determine the x range of the
            evaluated data if evaluate_function is set. If
            either of the limits are set to None, then the
            min/max value from the input data is used

        - evaluate_n is the number of samples to use when
            evaluating the function


        Returns:
            p, p_error, |residuals|, |x_fit|, |y_fit|
        """

        # Ensure that the x and y arrays are numpy arrays
        x = np.array(x)
        y = np.array(y)
        if y_err is not None:
            y_err = np.array(y_err)

        # Strip out any NaNs of Infs from the data
        valid_indices = np.logical_and(np.isfinite(x), np.isfinite(y))
        x = x[valid_indices]
        y = y[valid_indices]
        if y_err is not None:
            y_err = y_err[valid_indices]

        # Strip out any data-points that lie outside of the
        # fitting limits
        valid_indices = np.where((x_limit[0] <= x)
                                 & (x <= x_limit[1])
                                 & (y_limit[0] <= y)
                                 & (y <= y_limit[1]))
        x = x[valid_indices]
        y = y[valid_indices]
        if y_err is not None:
            y_err = y_err[valid_indices]

        # Define the fit parameter object
        p = FitParameters(self.parameter_names, constants, initialise)

        # Call the parameter initialiser
        if self.parameter_initialiser is not None:
            p.initialisation_mode = True
            self.parameter_initialiser(x, y, p)
            p.initialisation_mode = False

        # Populate the list of initial parameter values to be passed
        # to the fitting algorithm, and build list of bounds.
        p_init_list = []
        lower_bounds = []
        upper_bounds = []
        for name in p.variable_names:
            p_init_list.append(p[name])
            l, u = self.parameter_bounds.get(name, (-np.inf, np.inf))
            lower_bounds.append(l)
            upper_bounds.append(u)

        def LocalFitFunction(x, *p_list):
            # Transfer the parameters from the list used by the fit
            # algorithm to the fit parameter object
            p.absorb_variable_values(p_list)

            # Call the real fit function
            return self.fitting_function(x, p)

        # Execute the fitting algorithm
        # If an exception occurs, raise it as a FitError
        try:
            p_list, p_list_covariance = curve_fit(LocalFitFunction,
                x, y, p_init_list, sigma=y_err, absolute_sigma=True,
                bounds=(lower_bounds, upper_bounds), 
                x_scale='jac', method='trf')
        except Exception as e:
            raise FitError(e)

        # Now calculate the standard errors
        if np.any(np.isinf(p_list_covariance)):
            # If any of the covariances are infinite, return
            # a null array
            p_list_errors = np.zeros(len(p_list))
        else:
            p_list_errors = np.sqrt(np.diag(p_list_covariance))

        # Create dictionaries of the output fit parameters and errors
        p_dict = {}
        p_error_dict = {}

        # Copy the variable values
        for i in range(len(p.variable_names)):
            p_dict[p.variable_names[i]] = p_list[i]
            p_error_dict[p.variable_names[i]] = p_list_errors[i]

        # Copy the constants
        for name in p.constant_parameter_names:
            p_dict[name] = constants[name]
            p_error_dict[name] = 0

        # Calculate any derived parameters
        if self.derived_parameter_function is not None:
            p_dict, p_error_dict = \
                self.derived_parameter_function(p_dict, p_error_dict)

        if calculate_residuals:
            residuals = y - self.fitting_function(x, p_dict)

            # Now make sure residuals is the same length as the
            # original x array
            # TODO

        # If we don't need to evaluate the function
        if not evaluate_function:
            if calculate_residuals:
                return p_dict, p_error_dict, residuals
            else:
                return p_dict, p_error_dict

        # If we need to evaluate the function
        # Calculate the limits
        if evaluate_x_limit[0] is not None:
            x_limit_low = evaluate_x_limit[0]
        else:
            x_limit_low = np.min(x)

        if evaluate_x_limit[1] is not None:
            x_limit_high = evaluate_x_limit[1]
        else:
            x_limit_high = np.max(x)

        x_fit = np.linspace(x_limit_low, x_limit_high, evaluate_n)
        y_fit = self.fitting_function(x_fit, p_dict)

        if calculate_residuals:
            return p_dict, p_error_dict, residuals, x_fit, y_fit
        else:
            return p_dict, p_error_dict, x_fit, y_fit
