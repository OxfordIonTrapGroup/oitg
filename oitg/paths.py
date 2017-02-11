import os
from datetime import date


def _get_user():
    try:
        user = os.environ["OITG_USER"]
    except KeyError:
        raise Exception("No user supplied, and no OITG_USER environment key")
    return user


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


def analysis_root_path(user=None):
    """Returns the path to the given users analysis directory on the shared
    area. If the user is None, uses the enviroment variable 'OITG_USER'"""
    if user is None:
        user = _get_user()
    return os.path.join( shared_dir_path(), 'Users', user, 'analysis' )


def todays_analysis_path(day=None, user=None):
    """Returns the path to the analysis directory for given day, or todays
    date if day=None. Dates are in ISO format (yyyy-mm-dd). If the directory
    does not exist it is created. If user is None, uses the environment variable
    'OITG_USER'"""
    if day is None:
        day = date.today().isoformat()
    if user is None:
        user = _get_user()

    path = os.path.join(analysis_root_path(user=user), day)
    
    if not os.access(path, os.R_OK):
        # If the dir does not exist, create it
        os.mkdir(path)
    return path


def artiq_results_path(experiment=None):
    """Returns the path to an experiments Artiq results directory.
    'experiment' is the results subdirectory name. If None, the enviroment 
    variable 'OITG_EXPERIMENT' is used.

    The standard results path is <shared_area>/artiqResults/<experiment>"""

    path = os.path.join( shared_dir_path(), 'artiqResults' )

    if experiment is None:
        try:
            experiment = os.environ["OITG_EXPERIMENT"]
        except KeyError:
            raise Exception("No experiment supplied, and no OITG_EXPERIMENT environment key")
    path = os.path.join(path, experiment)

    return path
