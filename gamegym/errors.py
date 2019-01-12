class GameGymException(Exception):
    """
    Base class for GameGym exceptions (some informative/recoverable).
    """
    pass


class LimitExceeded(GameGymException):
    """
    Indicates that an algorithm with limit on e.g. visited nodes exceeded the limit.
    """
    pass


class GameGymError(GameGymException):
    """
    Base class for GameGym errors (generally unrecoverable).
    """
    pass
