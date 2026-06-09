from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        items=sorted(list(set(text)))
        itos={i:items[i] for i in range(len(items))}
        stoi={items[i]:i for i in range(len(items))}
        return stoi,itos
    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        res=[stoi[c] for c in text]
        return res 
    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        res=[itos[c] for c in ids]
        return "".join(res)
