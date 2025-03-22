from flask import (
    Flask,
    request,
    jsonify,
    g,
    redirect,
    url_for,
    Blueprint,
    make_response,
    render_template
)
from werkzeug.utils import safe_join, secure_filename  # secure_filename moved here
# Remove the import from security module
from werkzeug.security import generate_password_hash, check_password_hash  # If you need these
from datetime import datetime
import sys
import os
import requests
import base64
import binascii
import json
import logging
import csv
import tempfile
import re
import uuid
import base64
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from pydantic import BaseModel
from google.oauth2 import service_account
from flask_compress import Compress
from pydantic_settings import BaseSettings
from tenacity import retry, stop_after_attempt, wait_exponential
from functools import lru_cache
from ratelimit import limits, sleep_and_retry, RateLimitException
from openlibrary_search import fetch_books_from_openlibrary
from dotenv import load_dotenv
from flask_socketio import SocketIO, emit
# Add at the top of your file, after imports
import socket
# Set DNS resolution timeout and configure DNS
socket.setdefaulttimeout(20)  # 20 seconds timeout

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Decode base64-encoded Google credentials from Render
creds_b64 = os.getenv("GOOGLE_CREDENTIALS_B64")

if creds_b64:
    try:
        creds_bytes = base64.b64decode(creds_b64)
        creds_path = "/tmp/google-credentials.json"
        
        # Write to temp file
        with open(creds_path, "wb") as f:
            f.write(creds_bytes)

        # ✅ This is the key part: tell Google APIs where the file is
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

        credentials = service_account.Credentials.from_service_account_file(creds_path)

    except Exception as e:
        raise RuntimeError(f"Failed to decode or load credentials: {e}")
else:
    raise RuntimeError("Missing GOOGLE_CREDENTIALS_B64 in environment!")

# Force usage of system DNS resolver instead of eventlet's
os.environ['EVENTLET_NO_GREENDNS'] = 'yes'

# Enable mock data by default in development mode
USE_MOCK_DATA = os.environ.get("USE_MOCK_DATA", "false").lower() == "true"

# Define BookResponse model
class BookResponse(BaseModel):
    title: str
    authors: list[str]
    description: str | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "title": "The Great Gatsby",
                "authors": ["F. Scott Fitzgerald"],
                "description": "A story of the American dream...",
                "categories": ["Fiction", "Classic"],
                "publisher": "Scribner"
            }
        }
    }
# Define Settings model
class Settings(BaseSettings):
    """
    Configuration settings for the application.

    Attributes:
        GOOGLE_BOOKS_API_KEY (str): API key for accessing the Google Books API.
        GOOGLE_APPLICATION_CREDENTIALS (str): Path or content of Google service account credentials.
        MAX_RETRIES (int): Maximum number of retries for API requests. Default is 3.
        CACHE_TIMEOUT (int): Cache timeout duration in seconds. Default is 3600 seconds (1 hour).
        OPENAI_API_KEY (str): API key for OpenAI.
        SECRET_KEY (str): Secret key for the application.
        GOOGLE_DRIVE_FOLDER_ID (str): Google Drive folder ID for uploads.
        API_KEY (str): General API key for the application.
    """
    GOOGLE_BOOKS_API_KEY: str

    @property
    def validated_google_books_api_key(self) -> str:
        """Validate and return the stripped Google Books API key."""
        key = self.GOOGLE_BOOKS_API_KEY.strip()
        if not key:
            raise ValueError("GOOGLE_BOOKS_API_KEY is not set or is empty.")
        if not re.match(r'^[A-Za-z0-9_\-]+$', key):
            raise ValueError("GOOGLE_BOOKS_API_KEY is invalid. Please provide a valid API key.")
        return key
    MAX_RETRIES: int = 3
    CACHE_TIMEOUT: int = 3600
    OPENAI_API_KEY: str | None = None  # Make it optional
    SECRET_KEY: str
    GOOGLE_DRIVE_FOLDER_ID: str
    API_KEY: str

    class Config:
        env_file = '.env'
        extra = "allow"  # Allow extra fields

# Initialize settings
def get_settings():
    return Settings()
# Removed redundant register_routes function definition
# Initialize Blueprint
api_v1 = Blueprint('api_v1', __name__, url_prefix='/api/v1')

# Set up logging - place this near the top of the file
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define function to register routes
def register_routes(api_v1):
    """Registers all the routes for the api_v1 blueprint."""
    pass

# Define create_app function
def create_app():
    """Application factory pattern"""
    app = Flask(__name__)
    settings = get_settings()

    configure_app(app, settings)
    initialize_extensions(app)
    setup_routes(app)

    return app, settings

def configure_app(app, settings):
    """Configure the Flask app with settings."""
    app.config.update(
        GOOGLE_BOOKS_API_KEY=settings.validated_google_books_api_key,
        MAX_RETRIES=settings.MAX_RETRIES,
        CACHE_TIMEOUT=settings.CACHE_TIMEOUT
    )
    app.config['RESULTS_DIR'] = os.path.join(os.getcwd(), "learning", "Results")
    os.makedirs(app.config['RESULTS_DIR'], exist_ok=True)

def initialize_extensions(app):
    """Initialize Flask extensions."""
    compress = Compress()
    compress.init_app(app)

def setup_routes(app):
    """Set up routes and blueprints."""
    # API endpoints
    
    @app.route("/api")
    def api_index():
        """Serve the LibraryCloud API interface"""
        return render_template('api_s_index.html')
        
    @app.route('/home')
    def home():
        return redirect(url_for('home_page'))
    
    # Book search endpoints
    @app.route("/search_books")
    def search_books():
        """Search books endpoint with enhanced metadata extraction and mock data fallback."""
        query = request.args.get("query")
        if not query:
            return jsonify({"error": "Query parameter is required"}), 400
    
        try:
            # Try to fetch real data with a short timeout
            mock_param = request.args.get("mock", "")
            use_mock = (mock_param.lower() == "true" if mock_param else USE_MOCK_DATA)
            
            if (use_mock):
                # Use mock data if explicitly requested
                books = get_mock_books(query, "google")
                return jsonify({
                    "message": f"Found {len(books)} mock books for '{query}'",
                    "books": books,
                    "drive_link": None,
                    "mock": True
                })
            
            # Try to fetch real data
            books = fetch_books_from_google(query)
            
            if not books:
                # Fallback to mock if no real books found
                books = get_mock_books(query, "google")
                return jsonify({
                    "message": f"No real books found. Returning {len(books)} mock books for '{query}'",
                    "books": books,
                    "drive_link": None,
                    "mock": True
                })
                
            # Upload results to Google Drive
            file_path = upload_search_results_to_drive(books, query)
            return jsonify({
                "message": f"Found {len(books)} books for '{query}'",
                "books": books,
                "drive_link": file_path,
                "mock": False
            })
        except Exception as e:
            # Always fallback to mock data on error
            mock_books = get_mock_books(query, "google")
            return jsonify({
                "message": f"API error. Returning {len(mock_books)} mock books for '{query}'",
                "books": mock_books,
                "drive_link": None,
                "mock": True,
                "error": str(e)
            })

    # Other API routes...
    
    # Chat interface route
    @app.route('/')
    def home_page():
        """Serve the main Illustrator Co-Pilot interface"""
        return render_template('index.html')
    # Removed duplicate route definition for '/'
    app.register_blueprint(api_v1)

    # Register core routes
    @app.route("/api/welcome")
    def api_welcome():
        return "<h1>Welcome to the LibraryCloud API!</h1>"

    @app.route("/test_drive")
    def test_drive():
        try:
            service = get_drive_service()
            about = service.about().get(fields="user,storageQuota").execute()
            return jsonify({
                "success": True,
                "user": about.get("user", {}),
                "quota": about.get("storageQuota", {})
            })
        except Exception as e:
            logger.error(f"Drive test failed: {str(e)}", exc_info=True)
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500

    @app.route("/verify_credentials")
    def verify_credentials():
        """Verify Google Drive credentials configuration."""
        try:
            creds = app.config['GOOGLE_APPLICATION_CREDENTIALS']
            # Try to decode if it's base64
            try:
                decoded = base64.b64decode(creds).decode('utf-8') if creds else None
            except:
                decoded = None
            
            return jsonify({
                "success": True,
                "credentials_set": bool(creds),
                "credentials_type": str(type(creds)),
                "credentials_length": len(str(creds)) if creds else 0,
                "credentials_preview": str(creds)[:50] + "..." if creds else None,
                "decoded_preview": str(decoded)[:50] + "..." if decoded else None
            })
        except Exception as e:
            logger.error(f"Credential verification failed: {str(e)}", exc_info=True)
            return jsonify({"error": f"Credentials verification failed: {str(e)}"}), 500

    @app.route("/api/status")
    def api_status():
        """Show API status including mock mode"""
        return jsonify({
            "status": "ok",
            "mock_mode": USE_MOCK_DATA,
            "env": app.config.get("FLASK_ENV", "production"),
            "apis": {
                "google_books": bool(app.config.get("GOOGLE_BOOKS_API_KEY")),
                "google_drive": bool(app.config.get("GOOGLE_APPLICATION_CREDENTIALS")),
                "openai": bool(settings.OPENAI_API_KEY)
            }
        })

# Create app instance
app, settings = create_app()

# Initialize SocketIO
# Set up application configuration for Google credentials if not already set
if 'GOOGLE_APPLICATION_CREDENTIALS' not in app.config:
    app.config['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# Define SocketIO message handler
@socketio.on('message')
def handle_message(data):
    message = data.get('message', '')
    # Process the message and generate a response
    response = process_user_message(message)
    emit('response', {'data': response})

def process_user_message(message):
    # Simple fallback if OpenAI API is not available
    if not settings.OPENAI_API_KEY:
        if "process" in message.lower():
            return "Here are some process tips for Adobe Illustrator..."
        elif "design" in message.lower():
            return "Here are some design principles to consider..."
        elif "tutorial" in message.lower():
            return "Check out these Illustrator tutorials..."
        else:
            return "I'm here to help with Illustrator. Ask me about processes, design, or tutorials!"
    else:
        # Use OpenAI API for more advanced responses
        # Your existing OpenAI integration code here
        pass

def extract_book_info(item):
    """Extract book information including ISBN from API response."""
    volume_info = item.get('volumeInfo', {})
    
    # Add ISBN validation
    def validate_isbn(isbn):
        return isbn and re.match(r'^[\dX]{10,13}$', isbn.strip())
    
    # Extract and validate ISBNs
    isbns = []
    industry_identifiers = volume_info.get('industryIdentifiers', [])
    for identifier in industry_identifiers:
        isbn = identifier.get('identifier')
        if identifier.get('type') in ['ISBN_10', 'ISBN_13'] and validate_isbn(isbn):
            isbns.append(isbn)
    
    # Log ISBN extraction
    logger.info(f"Found {len(isbns)} valid ISBNs for book: {volume_info.get('title')}")
    
    return {
        'title': volume_info.get('title', 'Unknown Title'),
        'authors': volume_info.get('authors', []),
        'description': volume_info.get('description', ''),
        'isbn': isbns[0] if isbns else None,  # Use first ISBN found
        'isbn_10': next((i for i in isbns if len(i) == 10), None),
        'isbn_13': next((i for i in isbns if len(i) == 13), None),
        'publisher': volume_info.get('publisher', ''),
        'published_date': volume_info.get('publishedDate', ''),
        'page_count': volume_info.get('pageCount', 0),
        'categories': volume_info.get('categories', []),
        'language': volume_info.get('language', ''),
        'preview_link': volume_info.get('previewLink', ''),
        'info_link': volume_info.get('infoLink', '')
    }

@app.route("/list_results")
def list_results():
    """List saved search results."""
    try:
        # Fix: Use app.config to get RESULTS_DIR
        results_dir = app.config['RESULTS_DIR']
        if not os.path.exists(results_dir):
            logger.warning(f"Results directory does not exist: {results_dir}")
            return jsonify({
                "error": "Results directory does not exist",
                "directory": results_dir
            }), 404

        # List all CSV files in the results directory
        result_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
        return jsonify({
            "files": result_files,
            "count": len(result_files),
            "directory": results_dir
        })
    except Exception as e:
        logger.exception(f"Error listing results: {str(e)}")
        return jsonify({"error": "An unexpected error occurred while listing results"}), 500

@app.route("/get_file")
def get_file():
    filename = request.args.get("filename")
    if not filename:
        return jsonify({"error": "Filename parameter is required"}), 400

    results_dir = app.config['RESULTS_DIR']
    logger.info(f"Looking for file {filename} in directory {results_dir}")

    try:
        # Validate filename
        if not re.match(r'^[a-zA-Z0-9_.-]+$', filename):
            filename = secure_filename(filename)
            if not filename:
                return jsonify({"error": "Invalid filename format"}), 400
            
        filepath = os.path.join(results_dir, filename)
        
        # Check if file exists
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return jsonify({"error": "File not found"}), 404
            
        # Prevent directory traversal
        if not os.path.abspath(filepath).startswith(os.path.abspath(results_dir) + os.sep):
            logger.error(f"Security issue: Attempted to access file outside results directory")
            return jsonify({"error": "Security error"}), 403

        # Log file size
        file_size = os.path.getsize(filepath)
        logger.info(f"Serving file {filepath} with size {file_size} bytes")
        
        # Read file in binary mode
        with open(filepath, 'rb') as f:
            response = make_response(f.read())
            response.headers['Content-Type'] = 'text/csv'
            response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
            return response

    except Exception as e:
        logger.exception(f"Error serving file {filename}: {e}")
        return jsonify({"error": f"Error serving file: {str(e)}"}), 500

@app.route("/search_openlibrary")
def search_openlibrary():
    """Search OpenLibrary API endpoint with fallback to mock data."""
    query = request.args.get("query")
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    try:
        books = fetch_books_from_openlibrary(query)
        
        if not books and (app.config.get("FLASK_ENV") == "development" or request.args.get("mock") == "true"):
            # If no books found and in development or mock mode, return mock data
            books = get_mock_books(query, "openlibrary")
            
            return jsonify({
                "message": f"Found {len(books)} mock books (API connection failed)",
                "books": books,
                "drive_link": None,
                "mock": True,
            })
        
        # Rest of your function stays the same...
        
        # Upload results to Google Drive
        file_path = upload_search_results_to_drive(books, query)

        return jsonify({
            "message": f"Found {len(books)} books for '{query}'",
            "books": books,
            "drive_link": file_path,
            "mock": False
        })
        
    except Exception as e:
        logger.error(f"Error processing OpenLibrary search: {e}", exc_info=True)
        
        if app.config.get("FLASK_ENV") == "development" or request.args.get("mock") == "true":
            # Return mock data if there's an error
            mock_books = get_mock_books(query, "openlibrary")
            return jsonify({
                "message": f"Found {len(mock_books)} mock books (API error: {str(e)})",
                "books": mock_books,
                "drive_link": None,
                "mock": True  # Changed to True
            })
            
        return jsonify({"error": str(e)}), 500

# Add this function for fallback mock data

def get_mock_books(query, source="unknown"):
    """Return mock book data for testing when external APIs are unavailable."""
    return [
        {
            "title": f"{source.title()} Book About {query.title()}",
            "authors": ["API Connection Error"],
            "description": f"This is mock data because the {source} API connection failed. Try again later."
        },
        {
            "title": f"Another {source.title()} Book About {query.title()}",
            "authors": ["API Connection Error"],
            "description": "Mock data for testing purposes."
        }
    ]

# Define before_request function
@app.before_request
def before_request():
    g.request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
    logger.info(f"Processing request {g.request_id}: {request.method} {request.path}")

# Define fetch_books_from_google function
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=20))
def fetch_books_from_google(query):
    """Fetch books from Google Books API with improved retry logic."""
    if not query or not isinstance(query, str) or len(query.strip()) == 0:
        raise ValueError("Query parameter must be a non-empty string.")

    from urllib.parse import quote
    url = f"https://www.googleapis.com/books/v1/volumes?q={quote(query)}&key={app.config['GOOGLE_BOOKS_API_KEY'].strip()}"
    
    # Add custom headers to help with DNS resolution
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/json'
    }
    
    try:
        # Use a longer timeout and custom headers
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        return [extract_book_info(item) for item in data.get("items", [])]
    except requests.RequestException as e:
        logger.error(f"Error fetching books from Google: {str(e)}")
        raise BookAPIError(f"Failed to fetch books: {str(e)}")
# Log application details
logger.info(f"Application root: {os.path.dirname(__file__)}")
logger.info(f"Running on Heroku: {bool(os.getenv('HEROKU'))}")

from flask import current_app

def get_drive_service():
    """Returns an authenticated Google Drive service object."""
    google_credentials = current_app.config['GOOGLE_APPLICATION_CREDENTIALS']
    if not google_credentials:
        logger.error("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set")
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS is not set")
    
    logger.info(f"Credential type: {type(google_credentials)}")
    logger.info(f"Credential length: {len(google_credentials)}")
    
    try:
        # First, try to decode base64
        try:
            logger.info("Attempting base64 decode...")
            # Remove any whitespace/newlines that might have been added
            cleaned_creds = google_credentials.strip()
            creds_json = base64.b64decode(cleaned_creds).decode('utf-8')
            logger.info("Successfully decoded base64")
            credentials_info = json.loads(creds_json)
            logger.info("Successfully parsed JSON after base64 decode")
        except (binascii.Error, json.JSONDecodeError) as e:
            logger.warning(f"Base64 decode failed: {str(e)}")
            # If base64 fails, try direct JSON parsing
            try:
                logger.info("Attempting direct JSON parse...")
                credentials_info = json.loads(google_credentials)
                logger.info("Successfully parsed JSON directly")
            except json.JSONDecodeError as e:
                logger.error(f"All parsing attempts failed: {str(e)}")
                raise GoogleDriveError(f"Could not parse credentials: {str(e)}")

        # Create credentials object with specific scope
        credentials = service_account.Credentials.from_service_account_info(
            credentials_info,
            scopes=["https://www.googleapis.com/auth/drive.file"]
        )

        # Build and return the service
        service = build('drive', 'v3', credentials=credentials)
        logger.info("Successfully created Google Drive service")
        return service

    except Exception as e:
        logger.error(f"Failed to create Drive service: {str(e)}", exc_info=True)
        raise GoogleDriveError(f"Drive service creation failed: {str(e)}")

# Define custom exceptions
class GoogleDriveError(Exception):
    """Custom exception for Google Drive operations"""
    pass

class BookAPIError(Exception):
    """Custom exception for Google Books API operations"""
    pass

# Define upload_to_google_drive function
def upload_to_google_drive(file_path, file_name):
    """Uploads a file to Google Drive with enhanced logging."""
    if not os.path.exists(file_path):
        logger.error(f"File not found at path: {file_path}")
        raise GoogleDriveError(f"File not found: {file_path}")

    logger.info(f"Starting upload process for file: {file_name}")
    logger.info(f"File path: {file_path}")
    logger.info(f"File size: {os.path.getsize(file_path)} bytes")

    try:
        logger.info("Getting Drive service...")
        service = get_drive_service()
        
        logger.info("Creating file metadata with parent folder...")
        file_metadata = {
            'parents': [settings.GOOGLE_DRIVE_FOLDER_ID]  # Use the folder ID from settings
        }
        
        logger.info("Creating MediaFileUpload object...")
        media = MediaFileUpload(file_path, resumable=True)
        
        logger.info("Executing file creation...")
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        file_id = file.get('id')

        if not file_id:
            logger.error("No file ID received after upload")
            raise GoogleDriveError("Failed to get file ID after upload")

        logger.info(f"File uploaded successfully with ID: {file_id}")

        # Make the file publicly accessible
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
        def set_file_permissions():
            logger.info(f"Setting permissions for file ID: {file_id}")
            
            # Make file public
            public_permission = service.permissions().create(
                fileId=file_id,
                body={
                    "role": "reader",  # Changed from writer to reader: This allows anyone with the link to view the file, which is safer than granting edit permissions to the public.
                    "type": "anyone"
                }
            ).execute()
            logger.info(f"Public permission result: {public_permission}")

            # Add specific user permission
            user_permission = service.permissions().create(
                fileId=file_id,
                body={
                    "role": "writer",
                    "type": "user",
                    "emailAddress": os.environ.get("DRIVE_NOTIFICATION_EMAIL", "iwasonamountian@gmail.com")
                },
                sendNotificationEmail=False
            ).execute()
            logger.info(f"User permission result: {user_permission}")

        try:
            set_file_permissions()
            share_link = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
            logger.info(f"File shared successfully. Link: {share_link}")
            return share_link
        except Exception as e:
            logger.error(f"Error setting file permissions: {str(e)}", exc_info=True)
            # Return the link even if permission setting fails
            return f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"

    except Exception as e:
        logger.error(f"Error in upload_to_google_drive: {str(e)}", exc_info=True)
        return None

# Fix: A new function to handle both saving to CSV and uploading to Drive
def upload_search_results_to_drive(books, query):
    """Saves books to a temporary CSV file and uploads to Google Drive."""
    if not books:
        logger.warning("No books found for the given query.")
        return None

    temp_file = None
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp:
            temp_file = temp.name
        
        file_name = f'search_results_{query.replace(" ", "_")}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        # Write data to temp file
        with open(temp_file, "w", newline="", encoding="utf-8") as file:
            fieldnames = ["title", "authors", "description"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for book in books:
                book_row = {field: book.get(field, "") for field in fieldnames}
                if isinstance(book_row.get('authors'), list):
                    book_row['authors'] = ', '.join(book_row['authors'])
                writer.writerow(book_row)  # Write only once
        
        # Upload to Google Drive
        return upload_to_google_drive(temp_file, file_name)
        
    except Exception as e:
        logger.error(f"Error in upload_search_results_to_drive: {e}", exc_info=True)
        return None
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_file}: {e}", exc_info=True)
def validate_port(port_str):
    if not port_str.isdigit():
        raise ValueError(f"Invalid PORT environment variable: {port_str}. Must be a numeric value.")
    port = int(port_str)
    if port <= 0 or port > 65535:
        raise ValueError("Port number must be between 1 and 65535.")
    return port

# Application startup code
if __name__ == "__main__":
    port_env = os.environ.get("PORT", "5000")
    try:
        port = validate_port(port_env)
        # Use socketio.run instead of app.run
        is_debug_mode = os.environ.get("FLASK_ENV") == "development"
        socketio.run(app, host="0.0.0.0", port=port, use_reloader=is_debug_mode)
    except (ValueError, RuntimeError) as e:
        logger.error(f"Failed to start server: {e}")
# Move the application startup code to after all the implementation is done