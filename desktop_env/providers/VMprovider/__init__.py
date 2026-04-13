"""VMprovider - VM management and interaction for OpenGUIgym."""

from .client import VMClient
from .vm import VM, VMConfig

__all__ = ["VM", "VMClient", "VMConfig"]
