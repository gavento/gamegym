
class GameGymException(Exception):
    """
    Base class for GameGym exceptions.
    """
    pass


class LimitExceeded(GameGymException):
    """
    Indicates that an algorithm with limit on e.g. visited nodes exceeded the limit.
    """
    pass
