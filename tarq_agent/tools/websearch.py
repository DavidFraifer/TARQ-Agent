
from ..utils.console import console


class WebSearch:
    """Web search tool for retrieving information from the web."""

    def __init__(self, task_memory, user_input, task_id):
        
        self.task_memory = task_memory
        self.user_input = user_input
        self.task_id = task_id
        
        console.tool(f"[WEBSEARCH]", task_id=task_id)

        # Esto es lo que se ejecuta cuando llegue al modulo de search
        # Si quieres imprimir algo usa console.tool
        # Si quieres guardar algo en memoria usa self.task_memory.set()
        # La request del usuario la tienes en user_input
