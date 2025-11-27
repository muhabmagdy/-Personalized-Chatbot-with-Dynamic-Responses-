from typing import List, Optional
import os
from dotenv import load_dotenv
from serpapi import GoogleSearch
from langchain_core.documents import Document

load_dotenv()


class GoogleSearchRetriever:

    def __init__(self, api_key: Optional[str] = None, num_results: int = 5):

        self.api_key = api_key or os.environ.get("SERPAPI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "SERPAPI_API_KEY not found. "
                "Please add it to your .env file or pass it as a parameter."
            )
        
        self.num_results = num_results
    
    def search(self, query: str) -> List[str]:

        params = {
            "engine": "google",
            "q": query,
            "api_key": self.api_key,
            "num": self.num_results
        }
        
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            
            snippets = []
            for r in results.get("organic_results", []):
                snippet = r.get("snippet")
                if snippet:
                    snippets.append(snippet)
            
            return snippets
        
        except Exception as e:
            print(f"Google search error: {e}")
            return []
    
    def retrieve(self, query: str) -> List[Document]:

        params = {
            "engine": "google",
            "q": query,
            "api_key": self.api_key,
            "num": self.num_results
        }
        
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            
            documents = []
            for idx, r in enumerate(results.get("organic_results", [])):
                snippet = r.get("snippet", "")
                title = r.get("title", "")
                link = r.get("link", "")
                
                if snippet:
                   
                    content = f"{title}\n\n{snippet}"
                    
                    metadata = {
                        "source": "google_search",
                        "url": link,
                        "title": title,
                        "rank": idx + 1
                    }
                    
                    documents.append(Document(
                        page_content=content,
                        metadata=metadata
                    ))
            
            print(f"Retrieved {len(documents)} documents from Google Search")
            return documents
        
        except Exception as e:
            print(f"Google search error: {e}")
            return []
    
    def get_relevant_documents(self, query: str) -> List[Document]:
 
        return self.retrieve(query)
    
    def invoke(self, query: str) -> List[Document]:

        return self.retrieve(query)