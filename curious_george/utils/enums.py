from enum import Enum, EnumMeta

class MetaEnum(EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True


class AgentInputType(str, Enum, metaclass=MetaEnum):
    """How the agent observes. Both members are the pRNN's partial RGB view;
    they differ only in name and are kept because the questions repo passes
    `H_PO`. The seven visual/CANN members that used to sit beside them had no
    wrapper anyone constructed and were deleted 2026-09-06."""

    H_PO = "pRNN+PO"
    H = "pRNN"

class AgentType(str, Enum, metaclass=MetaEnum):
    RANDOM = "random"
    AC = "curious"
