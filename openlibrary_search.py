import httpx
import logging
import os
from typing import List, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

# Set up logging
logger = logging.getLogger(__name__)

# Check if we should use mock data
USE_MOCK_DATA = os.environ.get("USE_MOCK_DATA", "false").lower() == "true"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_books_from_openlibrary(query: str) -> List[Dict]:
    """
    Fetch books from Open Library API with improved retry logic and mock data fallback.
    
    Args:
        query (str): Search query string
        
    Returns:
        List[Dict]: List of books with title, authors, and description
    """
    # Return mock data if requested
    if USE_MOCK_DATA:
        logger.info(f"Using mock data for OpenLibrary query: {query}")
        return [
            {
                "title": f"OpenLibrary Book: {query.title()}",
                "authors": ["Mock Author 1", "Mock Author 2"],
                "description": f"This is mock data for {query} from OpenLibrary API."
            },
            {
                "title": f"The Complete Guide to {query.title()}",
                "authors": ["Library Expert"],
                "description": "A comprehensive mock resource from OpenLibrary."
            }
        ]
        
    from urllib.parse import quote
    encoded_query = quote(query)
    url = f"https://openlibrary.org/search.json?q={encoded_query}&limit=10"
    
    try:
        with httpx.Client(timeout=30) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()
        
            books = []
            for doc in data.get("docs", []):
                book = {
                    "title": doc.get("title", "Unknown Title"),
                    "authors": doc.get("author_name", []),
                    "description": None
                }
                
                # Try different fields for description in the order of priority:
                # "first_sentence" > "description" > "subtitle".
                # The first available field will be used as the description.
                for field in ["first_sentence", "description", "subtitle"]:
                    if field in doc:
                        value = doc[field]
                        if isinstance(value, list):
                            book["description"] = value[0]
                        else:
                            book["description"] = value
                        break
                books.append(book)
            
            logger.debug(f"Found {len(books)} books from OpenLibrary for query: {query}")
            logger.info(f"Found {len(books)} books from OpenLibrary for query: {query}")
            return books
            
    except httpx.RequestError as e:
        logger.error(f"Request error while fetching books from Open Library: {e}")
        return []
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP status error while fetching books from Open Library: {e}")
        return []
