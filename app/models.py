from abc import ABC, abstractmethod

from app.schemas import Listing


class RankingModel(ABC):
    @abstractmethod
    def rank(self, listings: list[Listing]) -> list[Listing]:
        ...


class ModelA(RankingModel):
    def rank(self, listings: list[Listing]) -> list[Listing]:
        return sorted(
            listings,
            key=lambda l: (l.rating * l.number_of_reviews) / (l.price + 1),
            reverse=True,
        )


class ModelB(RankingModel):
    def rank(self, listings: list[Listing]) -> list[Listing]:
        return sorted(
            listings,
            key=lambda l: l.rating - (l.price / 1000),
            reverse=True,
        )
