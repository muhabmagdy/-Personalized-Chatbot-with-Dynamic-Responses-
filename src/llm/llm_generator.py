from typing import List
import google.generativeai as genai


class LLMGenerator:
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.0-flash",
        max_tokens: int = 1000,
        temperature: float = 0.1,
    ):
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
        ]
        
        self.model = genai.GenerativeModel(model_name)

    def build_prompt(self, query: str, context: str) -> str:
        """Build the LLM prompt using query + context."""
        return f"""You are a senior Python mentor who adapts your role based on the student's request.

Your dynamic responsibilities:
1. If the student asks for an explanation → give a clear, beginner-friendly explanation with one small example.
2. If the student asks for debugging → identify issues, explain them, and provide a corrected version.
3. If the student asks for code review → evaluate readability, style, and correctness and give improvements.
4. If the student asks for solving a problem → break it down step-by-step and provide an optimal solution.
5. If the student asks for a concept → explain only that concept without expanding unnecessarily.
6. If the student provides incomplete information → ask a clarifying question instead of assuming details.
7. If the student asks for best practices → give concise recommendations and justify each one.

Global Guardrails:
- Base ALL answers ONLY on the student's message and the provided context.
- Do NOT hallucinate missing context or assume unknown requirements.
- If something is unclear, incomplete, ambiguous, or impossible to infer, ask for clarification.
- Keep examples short unless asked for long/full code.

Context:
{context}

Student Request:
{query}

Generate a helpful, safe, and accurate response.
"""

    def answer_generation(self, query: str, retrieved_docs: List[str], memory=None) -> str:
        """Generates an answer using RAG documents + optional long-term memory."""

        context = "\n\n".join(retrieved_docs)

        memory_context = ""
        if memory:
            try:
                mem_docs = memory.load_memory_variables({})
                memory_context = mem_docs.get("history", "")
            except Exception:
                memory_context = ""

        if memory_context:
            context = f"### Long-Term Memory:\n{memory_context}\n\n### Retrieved Knowledge (RAG):\n{context}"

        prompt = self.build_prompt(query, context)

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                },
                safety_settings=self.safety_settings
            )
 
            if not response.candidates:
                return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
            
            candidate = response.candidates[0]

            if candidate.finish_reason == 10:  
                return "I apologize, but the response was blocked due to safety filters. Please try rephrasing your question."
            
            if candidate.finish_reason == 10:  
                return "I apologize, but I couldn't generate a unique response. Please try a different question."
            
            if candidate.content and candidate.content.parts:
                return candidate.content.parts[0].text
  
            return response.text
            
        except ValueError as e:
            if "finish_reason" in str(e):
                return "I apologize, but I couldn't generate a response for this query. Please try rephrasing your question or ask something different."
            raise e
        except Exception as e:
            return f"Error generating response: {str(e)}"