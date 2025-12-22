# Copyright (c) 2025 TeleAI-infra Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union
import copy
import inspect

T = TypeVar('T')


class RegistryError(Exception):
    """Base exception for registry-related errors."""
    pass


class ModuleAlreadyRegisteredError(RegistryError):
    """Raised when attempting to register a module that already exists."""
    pass


class ModuleNotFoundError(RegistryError):
    """Raised when attempting to build a module that doesn't exist in registry."""
    pass


class Registry:
    """
    A registry for managing and instantiating modules dynamically.
    
    This registry allows you to register classes/functions and later instantiate
    them by name with configuration parameters.
    
    Example:
        >>> registry = Registry("processors")
        >>> 
        >>> @registry.register
        >>> class TextProcessor:
        >>>     def __init__(self, config: str):
        >>>         self.config = config
        >>> 
        >>> # or register with custom name
        >>> registry.register_module(TextProcessor, "custom_processor")
        >>> 
        >>> # Build instances
        >>> processor = registry.build("TextProcessor", config="my_config")
        >>> custom = registry.build({"type": "custom_processor", "config": "data"})
    """
    
    def __init__(self, name: str = "Registry"):
        """
        Initialize the registry.
        
        Args:
            name: A descriptive name for this registry (used in error messages).
        """
        self.name = name
        self._modules: Dict[str, Type] = {}
    
    def register_module(self, module_class: Type[T], module_name: Optional[str] = None) -> Type[T]:
        """
        Register a module class with the registry.
        
        Args:
            module_class: The class to register.
            module_name: Optional custom name. If None, uses class.__name__.
            
        Returns:
            The registered module class (for decorator chaining).
            
        Raises:
            ModuleAlreadyRegisteredError: If module_name already exists.
        """
        if module_name is None:
            module_name = module_class.__name__
            
        if module_name in self._modules:
            raise ModuleAlreadyRegisteredError(
                f"Module '{module_name}' is already registered in {self.name}. "
                f"Existing: {self._modules[module_name]}, "
                f"New: {module_class}"
            )
        
        self._modules[module_name] = module_class
        return module_class
    
    def register(self, name_or_class: Union[str, Type[T]]) -> Union[Callable[[Type[T]], Type[T]], Type[T]]:
        """
        Register a module class, supporting both direct registration and decorator usage.
        
        Usage:
            # Direct registration
            registry.register(MyClass)
            
            # Decorator without custom name
            @registry.register
            class MyClass: ...
            
            # Decorator with custom name
            @registry.register("custom_name")
            class MyClass: ...
        
        Args:
            name_or_class: Either a string name or the class to register.
            
        Returns:
            Either the registered class or a decorator function.
        """
        if isinstance(name_or_class, str):
            # Decorator with custom name
            def decorator(module_class: Type[T]) -> Type[T]:
                return self.register_module(module_class, name_or_class)
            return decorator
        else:
            # Direct registration or decorator without custom name
            return self.register_module(name_or_class)
    
    def unregister(self, module_name: str) -> None:
        """
        Remove a module from the registry.
        
        Args:
            module_name: Name of the module to remove.
            
        Raises:
            ModuleNotFoundError: If module doesn't exist.
        """
        if module_name not in self._modules:
            raise ModuleNotFoundError(f"Module '{module_name}' not found in {self.name}")
        del self._modules[module_name]
    
    def get_module(self, module_name: str) -> Type:
        """
        Get a registered module class by name.
        
        Args:
            module_name: Name of the module to retrieve.
            
        Returns:
            The registered module class.
            
        Raises:
            ModuleNotFoundError: If module doesn't exist.
        """
        if module_name not in self._modules:
            available = list(self._modules.keys())
            raise ModuleNotFoundError(
                f"Module '{module_name}' not found in {self.name}. "
                f"Available modules: {available}"
            )
        return self._modules[module_name]
    
    def build(self, name: str,config=None, *args, **kwargs) -> Any:
        """
        Build and instantiate a registered module by name.

        Args:
            name: The name of the registered module to build.
            *args: Positional arguments to pass to the module constructor.
            **kwargs: Keyword arguments to pass to the module constructor.

        Returns:
            An instance of the requested module.

        Raises:
            ModuleNotFoundError: If the specified module doesn't exist.
            TypeError: If the module constructor fails.
        """
        module_class = self.get_module(name)

        try:
            return module_class(config=config, *args, **kwargs)
        except TypeError as e:
            # Enhance error message with signature info
            sig = inspect.signature(module_class.__init__)
            raise TypeError(
                f"Failed to instantiate {name}: {e}. "
                f"Expected signature: {sig}"
            ) from e
    
    
    def _normalize_config(self, config: Union[str, Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Normalize different config input formats into a consistent dict.
        
        Args:
            config: String type name or dict configuration.
            **kwargs: Additional parameters to merge.
            
        Returns:
            Normalized configuration dictionary.
        """
        if isinstance(config, str):
            result = {"type": config}
        elif isinstance(config, dict):
            result = copy.deepcopy(config)
        else:
            raise ValueError(f"Config must be string or dict, got {type(config)}")
        
        # Merge kwargs, checking for conflicts
        for key, value in kwargs.items():
            if key in result:
                raise ValueError(f"Parameter '{key}' specified in both config and kwargs")
            result[key] = value
        
        return result
    
    def list_modules(self) -> Dict[str, Type]:
        """
        Get a copy of all registered modules.
        
        Returns:
            Dictionary mapping module names to their classes.
        """
        return self._modules.copy()
    
    def __contains__(self, module_name: str) -> bool:
        """Check if a module is registered."""
        return module_name in self._modules
    
    def __len__(self) -> int:
        """Get the number of registered modules."""
        return len(self._modules)
    
    def __repr__(self) -> str:
        """String representation of the registry."""
        modules = list(self._modules.keys())
        return f"{self.name}({len(modules)} modules: {modules})"





# Example usage and tests
if __name__ == "__main__":
    # Create a registry
    registor = Registry("MyRegistry")
    
    from .wan.parallel_wan_model import ParallelWanModel
    from .teleai.parallel_teleai_model import ParallelTeleaiModel
    def build_model(name,config=None):
        if config is None:
            return registor.build(name)
        else:
            return registor.build(name,config)
    # Register using different methods
    registor.register(ParallelWanModel)
    registor.register(ParallelTeleaiModel)
    
    # Build instances
    # processor1 = registor.build(")
    
    # print(f"Processor1 config: {processor1.config}")