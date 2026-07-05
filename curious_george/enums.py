from enum import Enum, EnumMeta

class MetaEnum(EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True 


class AgentInputType(str, Enum, metaclass=MetaEnum):
    H_PO = "pRNN+PO"
    Visual_FO = "Visual_FO"
    Visual_PO = "Visual_PO"
    PC = "PC"
    CANN = "CANN"
    PC_PO = "PC+PO"
    H = "pRNN"
    CANN_PO = "CANN+PO"
    CANN_norecurr = "CANN_norecurrence"

class AgentType(str, Enum, metaclass=MetaEnum):
    RANDOM = "random"
    AC = "curious"
