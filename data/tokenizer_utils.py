from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        n=len(numbers)
        res=[[] for _ in range(n)]
        for j in range(n):
            word=numbers[j]
            char_arr=list(str(word))
            pos=0
            while pos< len(char_arr):
                longest=None
                for i in range(pos+1,len(char_arr)+1):
                    window_raw=char_arr[pos:i]
                    candidate="".join(window_raw)
                    if candidate in vocab:
                        longest=candidate
                res[j].append(longest)
                pos+=len(longest)
        return res
    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        count=0
        pos=0
        while pos<len(text):
            longest=None
            for i in range(pos+1,len(text)+1):
                token_candidate=text[pos:i]
                if token_candidate in vocab:
                    longest=token_candidate
            count+=1
            pos=pos+len(longest)
        return count

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        tokens = self.count_tokens(text, vocab)
        words = len(text.split())
        return round((tokens/words),4)
