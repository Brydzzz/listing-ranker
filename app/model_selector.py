import hashlib

from app.models import Model1, Model2, RankingModel

model1 = Model1()
model2 = Model2()


def get_model_for_user(user_id: str) -> tuple[RankingModel, str]:
    digest = hashlib.sha256(user_id.encode()).hexdigest()
    bucket = int(digest, 16) % 2
    if bucket == 0:
        return model1, "model1"
    return model2, "model2"
