import os
from datetime import date


def shared_dir_path():
    """Returns the standard path to the shared area for this platform.
    For Windows the standard mount point is Z:\
    For Linux / OSX the standard mount point is ~/steaneShared"""
    
    if os.name == 'nt': # Windows
        path = 'Z:\\'
    elif os.name == 'unix' or os.name == 'posix': # Linux / OSX / ...
        path = os.path.expanduser('~/steaneShared/')
    else:
        raise Exception('Unknown OS')
    
    return path


def analysis_root_path(user='Chris'):
    """Returns the path to the given users analysis directory on the shared
    area"""
    return os.path.join( shared_dir_path(), 'Users', user, 'analysis' )


def todays_analysis_path(day=None, user='Chris'):
    """Returns the path to the analysis directory for given day, or todays
    date if day=None. Dates are in ISO format (yyyy-mm-dd). If the directory
    does not exist it is created."""
    if day is None:
        day = date.today().isoformat()
    
    path = os.path.join(analysis_root_path(user=user), day)
    
    if not os.access(path, os.R_OK):
        # If the dir does not exist, create it
        os.mkdir(path)
    return path


def artiq_results_path(experiment=None):
    """Returns the path to the standard Artiq results directory.
    If experiment is None, returns the path to the master results directory 
    containing the experiment subdirectories.
    Else appends 'experiment' to the path, returning the path to a given 
    experiment's results.

    The standard results path is <shared_area>/artiqResults/<experiment>"""

    path = os.path.join( shared_dir_path(), 'artiqResults' )

    if experiment is not None:
        path = os.path.join(path, experiment)

    return path
