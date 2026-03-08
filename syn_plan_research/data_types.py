from dataclasses import dataclass, field
from typing import Literal, Dict, List
import json

@dataclass
class GenRecord:
    id: str 
    question: str
    golden_answers: List[str]
    data_source: str 
    prompt: str 
    response: str
    active: bool = field(default=True)
    
    llm_call_num: int = field(default=0)
    
    answer: str = field(default=None)
    em: float = field(default=None)
    
    # track history 
    actions: List = field(default_factory=list)


class Message:
    def __init__(self, role: Literal["user", "assistant", "system", "tool"], content: str):
        assert role in ("user", "assistant", "system", "tool"), f"Invalid role: {role}"
        assert isinstance(content, str), "Content must be a string"
        self.role = role
        self.content = content

    def to_dict(self) -> Dict:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, d: Dict):
        return cls(role=d["role"], content=d["content"])

    def copy(self):
        return Message(role=self.role, content=self.content)

    def __repr__(self):
        return f"Message(role={self.role!r}, content={self.content!r})"

    def __eq__(self, other):
        return isinstance(other, Message) and self.role == other.role and self.content == other.content

    def __hash__(self):
        return hash((self.role, self.content))
