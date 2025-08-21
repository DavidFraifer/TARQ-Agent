class TaskMemory:
    def __init__(self, name: str, max_lines: int = 50):
        self.name = name
        self._lines = []
        self.max_lines = max_lines

    def set(self, value: str):
        self._lines.append(str(value))
        
        if len(self._lines) > self.max_lines:
            excess = len(self._lines) - self.max_lines
            self._lines = self._lines[excess:]
            
            if not self._lines[0].startswith("[Memory truncated"):
                self._lines[0] = f"[Memory truncated - showing last {self.max_lines} entries]"

    def get(self) -> str:
        return "\n".join(f"{i+1}. {line}" for i, line in enumerate(self._lines)) if self._lines else ""
    
    def get_line_count(self) -> int:
        return len(self._lines)