from abc import ABC, abstractmethod
from typing import Any

from cloth_opt.sim.action import ClothAction


class BasePolicy(ABC):
    def reset(self) -> None:
        pass

    @abstractmethod
    def get_action(self, observation: dict[str, Any]) -> tuple[ClothAction, dict[str, Any]] | None:
        raise NotImplementedError

    def close(self) -> None:
        pass
