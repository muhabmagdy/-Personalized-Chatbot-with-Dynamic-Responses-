from typing import List, Dict
from .answer_relevance import score_answer_relevance
from .context_relevance import score_context_relevance
from .groundedness import score_groundedness


def evaluate_answer(
    question: str,
    answer: str,
    reference_answers: List[str] = None,
    contexts: List[str] = None,
    sources: List[str] = None
) -> Dict[str, float]:
    """
    Evaluates the answer using multiple metrics.
    
    Args:
        question: The user's question
        answer: The generated answer
        reference_answers: List of expected/correct answers (optional)
        contexts: Retrieved context chunks used to generate the answer
        sources: Source documents
    
    Returns:
        Dictionary with evaluation scores (0.0 to 1.0)
        {
            'answer_relevance': float,
            'context_relevance': float,
            'groundedness': float
        }
    """
    scores = {}
  
    scores['answer_relevance'] = score_answer_relevance(
        question=question,
        answer=answer,
        reference_answers=reference_answers or []
    )
    

    if contexts:
        scores['context_relevance'] = score_context_relevance(
            answer=answer,
            contexts=contexts
        )
    else:
        scores['context_relevance'] = 0.0
    

    if sources:
        scores['groundedness'] = score_groundedness(
            answer=answer,
            sources=sources
        )
    else:
        scores['groundedness'] = 0.0
    
    return scores