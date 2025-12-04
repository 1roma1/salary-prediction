from typing import Any, Callable


class BaseRegistry:
    @classmethod
    def register(cls, name: str) -> Callable:
        def inner_wrapper(wrapped_class: Any) -> Any:
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def get(cls, name: str) -> Any:
        if name in cls.registry:
            return cls.registry[name]
        else:
            raise ValueError(
                f"There is no element: {name}, "
                f"available: {cls.registry.keys()}"
            )


class ModelRegistry(BaseRegistry):
    registry: dict[str, Any] = {}


class LossRegistry(BaseRegistry):
    registry: dict[str, Any] = {}


class OptimizerRegistry(BaseRegistry):
    registry: dict[str, Any] = {}
