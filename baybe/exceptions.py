"""Custom exceptions."""


class NotEnoughPointsLeftError(Exception):
    """
    More recommendations are requested than there are viable parameter configurations
    left in the search space.
    """


class NoMCAcquisitionFunctionError(Exception):
    """
    A Monte Carlo acquisition function is required but an analytical acquisition
    function has been selected by the user.
    """


class IncompatibleSearchSpaceError(Exception):
    """
    A recommender is used with a search space that contains incompatible parts,
    e.g. a discrete recommender is used with a hybrid or continuous search space.
    """


class EmptySearchSpaceError(Exception):
    """The created search space contains no parameters."""


class NothingToSimulateError(Exception):
    """There is nothing to simulate because there are no testable configurations."""


class NoRecommendersLeftError(Exception):
    """A recommender is requested by a meta recommender but there are no recommenders
    left.
    """


class NumericalUnderflowError(Exception):
    """A computation would lead to numerical underflow."""


class OptionalImportError(Exception):
    """An attempt was made to import an optional but uninstalled dependency."""


class DeprecationError(Exception):
    """Signals the use of a deprecated mechanism to the user, interrupting execution."""


class UnidentifiedSubclassError(Exception):
    """A specified subclass cannot be found in the given class hierarchy."""
