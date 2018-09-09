import vizdoom
import os


class Loader():
    """
    This class converts file name to full paths to be imported
    by the DoomGame
    """
    def get_vizdoom_path(self):
        package_directory = os.path.dirname(os.path.abspath(vizdoom.__file__))
        return os.path.join(package_directory, 'vizdoom')

    def get_freedoom_path(self):
        package_directory = os.path.dirname(os.path.abspath(vizdoom.__file__))
        return os.path.join(package_directory, 'freedoom2.wad')

    def get_scenario_path(self, name):
        package_directory = os.path.dirname(os.path.abspath(vizdoom.__file__))
        return os.path.join(package_directory, 'scenarios/{}'.format(name))
