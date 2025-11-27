from typing import List, Tuple


class ConversationBufferMemory:
 
    def __init__(self, window_size: int = 3):
 
        self.window_size = window_size
        self.exchanges: List[Tuple[str, str]] = []  

    def add_memory(self, user_input: str, ai_output: str):
        self.exchanges.append((user_input, ai_output))

        if len(self.exchanges) > self.window_size:
            self.exchanges = self.exchanges[-self.window_size:]

    def add_exchange(self, user_input: str, ai_output: str):
        """Alias for add_memory."""
        self.add_memory(user_input, ai_output)

    def get_memory(self) -> str:
        if not self.exchanges:
            return ""
        
        formatted = []
        for user_msg, ai_msg in self.exchanges:
            formatted.append(f"User: {user_msg}")
            formatted.append(f"Assistant: {ai_msg}")
        
        return "\n".join(formatted)

    def get_all_messages(self) -> str:
        mem = self.get_memory()
        return mem if mem else "No conversation history"

    def get_message_count(self) -> int:
        return len(self.exchanges) * 2

    def get_exchange_count(self) -> int:
        return len(self.exchanges)

    def clear(self):
        self.exchanges = []

    def clear_memory(self):
        self.clear()

    def get_context_for_prompt(self, query: str = "", include_search: bool = False) -> str:
        return self.get_memory()

    def should_search_web(self, query: str) -> bool:
        if not query:
            return False
        
        query_lower = query.lower()

        time_indicators = [
            "latest", "recent", "current", "news", "today", "update", 
            "now", "trending", "this year", "2024", "2025", "yesterday",
            "last week", "last month"
        ]
        
        return any(indicator in query_lower for indicator in time_indicators)