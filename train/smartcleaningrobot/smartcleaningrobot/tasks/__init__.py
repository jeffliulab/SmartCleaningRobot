"""Task definitions — each sub-package registers its own Gym environments."""

from isaaclab_tasks.utils import import_packages

_BLACKLIST_PKGS = ["utils", ".mdp", "_shared"]
import_packages(__name__, _BLACKLIST_PKGS)
