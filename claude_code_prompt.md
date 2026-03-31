# StyleMate — AI-Powered Multimodal Fashion Shopping Assistant

## Project Overview

An AI-powered shopping assistant called **StyleMate** for an online clothing store. Customers can search for clothing products using natural language text queries OR by uploading images of outfits they like. The system uses Google's Gemini Embedding model for multimodal embeddings (text + images) and Supabase as the vector database.

---

## Tech Stack

- **Backend:** Python 3.11+ with FastAPI
- **Database:** Supabase (PostgreSQL with pgvector extension)
- **Embeddings:** Google Gemini Embedding 2 Preview (`gemini-embedding-2-preview`) — 3072-dimensional vectors
- **LLM:** Google Gemini 2.5 Flash (`gemini-2.5-flash`) for generating conversational responses
- **Frontend:** Simple HTML + CSS + JavaScript (no framework — single page app)
- **API Client:** Google Generative AI Python SDK (`google-genai`)

---

## Project Structure

```
stylemate/
├── backend/
│   ├── main.py                 # FastAPI application with all routes
│   ├── ingestion.py            # Script to process products and store embeddings
│   ├── embedding_service.py    # Gemini Embedding helper functions
│   ├── search_service.py       # Vector search logic (text + image)
│   ├── chat_service.py         # Gemini Flash response generation
│   ├── supabase_client.py      # Supabase connection setup
│   └── requirements.txt        # Python dependencies
├── frontend/
│   ├── index.html              # Chat interface
│   ├── styles.css              # Styling
│   └── app.js                  # Frontend logic
├── data/
│   ├── products.csv            # Product catalog (10 products)
│   └── images/                 # Real product images (manually sourced)
├── .env                        # Environment variables (not committed)
└── claude_code_prompt.md       # This file
```

---

## Environment Variables

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_gemini_api_key_here
SUPABASE_URL=your_supabase_project_url_here
SUPABASE_SERVICE_KEY=your_supabase_service_role_key_here
```

---

## Database Structure

### Supabase Setup

Enable the `vector` extension in Supabase:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Products Table

```sql
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    product_id TEXT UNIQUE NOT NULL,
    product_name TEXT NOT NULL,
    category TEXT NOT NULL,
    color TEXT NOT NULL,
    size TEXT NOT NULL,
    price NUMERIC NOT NULL,
    rating NUMERIC NOT NULL,
    description TEXT NOT NULL,
    image_filename TEXT NOT NULL,
    product_link TEXT NOT NULL,
    text_embedding VECTOR(3072),
    image_embedding VECTOR(3072)
);
```

### Vector Search Functions

```sql
-- Text-based similarity search
CREATE OR REPLACE FUNCTION match_products_by_text(
    query_embedding VECTOR(3072),
    match_threshold FLOAT DEFAULT 0.5,
    match_count INT DEFAULT 6
)
RETURNS TABLE (
    id INT,
    product_id TEXT,
    product_name TEXT,
    category TEXT,
    color TEXT,
    size TEXT,
    price NUMERIC,
    rating NUMERIC,
    description TEXT,
    image_filename TEXT,
    product_link TEXT,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.id,
        p.product_id,
        p.product_name,
        p.category,
        p.color,
        p.size,
        p.price,
        p.rating,
        p.description,
        p.image_filename,
        p.product_link,
        1 - (p.text_embedding <=> query_embedding) AS similarity
    FROM products p
    WHERE 1 - (p.text_embedding <=> query_embedding) > match_threshold
    ORDER BY p.text_embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Image-based similarity search
CREATE OR REPLACE FUNCTION match_products_by_image(
    query_embedding VECTOR(3072),
    match_threshold FLOAT DEFAULT 0.5,
    match_count INT DEFAULT 6
)
RETURNS TABLE (
    id INT,
    product_id TEXT,
    product_name TEXT,
    category TEXT,
    color TEXT,
    size TEXT,
    price NUMERIC,
    rating NUMERIC,
    description TEXT,
    image_filename TEXT,
    product_link TEXT,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.id,
        p.product_id,
        p.product_name,
        p.category,
        p.color,
        p.size,
        p.price,
        p.rating,
        p.description,
        p.image_filename,
        p.product_link,
        1 - (p.image_embedding <=> query_embedding) AS similarity
    FROM products p
    WHERE 1 - (p.image_embedding <=> query_embedding) > match_threshold
    ORDER BY p.image_embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
```

---

## Product Catalog (products.csv)

10 products across categories: Polo T-Shirts, Formal Shirts, Jackets, T-Shirts, Shorts, Casual Shirts, Chinos, Kurtas, Jeans, Trousers.

| ID | Name | Image File |
|----|------|-----------|
| P001 | Classic Navy Blue Polo T-Shirt | `polo_navy.jpg` |
| P002 | Slim Fit Black Formal Shirt | `formal_black.jpg` |
| P003 | Washed Indigo Denim Jacket | `denim_jacket_indigo.jpg` |
| P004 | White Crew Neck T-Shirt | `tshirt_white.jpg` |
| P005 | Olive Green Cargo Shorts | `shorts_olive.jpg` |
| P006 | Sky Blue Casual Shirt | `casual_skyblue.jpg` |
| P007 | Charcoal Slim Fit Chinos | `chinos_charcoal.jpg` |
| P008 | Maroon Cotton Kurta | `kurta_maroon.jpg` |
| P009 | Dark Wash Slim Fit Jeans | `jeans_darkblue.jpg` |
| P010 | Beige Formal Trousers | `trousers_beige.jpg` |

**Product images** are real product photos (not generated). Source them from the internet or generate with an AI image tool, then place them in `data/images/`. Recommended size: ~400x500px, clean white/neutral background.

---

## Embedding Service (embedding_service.py)

### Text Embeddings

```python
from google import genai

client = genai.Client(api_key=GEMINI_API_KEY)

def get_text_embedding(text: str) -> list[float]:
    response = client.models.embed_content(
        model="gemini-embedding-2-preview",
        contents=text,
        config={
            "task_type": "SEMANTIC_SIMILARITY",
            "output_dimensionality": 3072
        }
    )
    return response.embeddings[0].values
```

### Image Embeddings

```python
from google.genai import types

def get_image_embedding(image_path: str) -> list[float]:
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    image_part = types.Part(
        inline_data=types.Blob(
            mime_type="image/jpeg",
            data=image_bytes
        )
    )

    response = client.models.embed_content(
        model="gemini-embedding-2-preview",
        contents=types.Content(parts=[image_part]),
        config={
            "task_type": "SEMANTIC_SIMILARITY",
            "output_dimensionality": 3072
        }
    )
    return response.embeddings[0].values
```

### Rate Limit Handling

- Wrap API calls in a retry decorator with exponential backoff (max 3 retries)
- Log progress during ingestion: "Processing product X of Y..."

---

## Ingestion Pipeline (ingestion.py)

1. Read `data/products.csv` using pandas
2. For each product:
   a. Combine text fields into a single rich string for embedding
   b. Generate a text embedding using `get_text_embedding()`
   c. Generate an image embedding from the product's image file in `data/images/`
   d. Upsert the product record into Supabase including both embeddings
3. Print progress and a summary at the end

Run with: `python backend/ingestion.py`

---

## Chat Service (chat_service.py)

Use Gemini 2.5 Flash (`gemini-2.5-flash`) to generate conversational responses with a fashion assistant system prompt.

---

## FastAPI Backend (main.py)

### Routes

```
POST /api/chat          — Handle text-only chat messages
POST /api/chat-image    — Handle messages with image upload
GET  /api/products      — List all products (for debugging)
GET  /                  — Serve the frontend
```

Serve `frontend/` as static files. Serve `data/images/` at `/images/`.

### Similarity Threshold

The minimum cosine similarity for a product to be included in results is **0.5** (set as `SIMILARITY_THRESHOLD` in `search_service.py`). Products that do not meet this threshold are excluded before the response is sent to the frontend — this prevents loosely-related categories (e.g. t-shirts appearing for a "shorts" query) from being shown.

Tune `SIMILARITY_THRESHOLD` in `search_service.py` if needed:
- Raise it (e.g. `0.55`) for stricter category matching
- Lower it (e.g. `0.45`) if too few results are returned for valid queries

### Image + Text Merge Logic

When both an image and a text message are provided to `/api/chat-image`, results from both searches are merged. If the same product appears in both sets, the higher similarity score is kept. The merged list is then sorted by similarity (descending) and truncated to 6 before being returned.

### Price Filtering

Price constraints are handled in `main.py` via three helper functions — **not** left to the LLM. This is necessary because budget-phrased queries (e.g. "clothes under ₹1000") produce embeddings that don't resemble clothing descriptions, so the vector search returns zero results without this fallback.

#### `extract_price_filter(message) -> (min_price, max_price)`

Parses the user message with regex to detect budget constraints. Returns a `(min_price, max_price)` tuple — either value is `None` if that bound was not mentioned.

Recognised phrasings:
| Intent | Examples |
|--------|---------|
| Upper bound | `under Rs1000`, `less than 1000`, `below 1000`, `up to 1500`, `maximum 2000`, `at most 999` |
| Lower bound | `above Rs500`, `more than 500`, `over 800`, `at least 1000`, `minimum 600`, `starting from 700` |
| Range | `between 500 and 1500`, `500 to 1500`, `Rs800 - Rs2000` |

#### `apply_price_filter(products, min_price, max_price) -> list`

Filters a product list to only include items whose `price` falls within the extracted bounds (both bounds inclusive). Applied after the vector search in both `/api/chat` and `/api/chat-image`.

#### `fetch_all_products() -> list`

Fetches the full product catalog from Supabase (no vector scoring). Used as a **fallback** in `/api/chat` when the vector search returns zero results but a price constraint was detected — ensuring budget-only queries like "show me everything under ₹1000" always return relevant products.

#### Request flow for `/api/chat` with a price query

```
User message
    │
    ├─ extract_price_filter()        → (min_price, max_price)
    │
    ├─ search_by_text()              → vector results (may be empty for budget queries)
    │
    ├─ [if empty + price constraint] → fetch_all_products()  ← fallback
    │
    ├─ apply_price_filter()          → price-filtered list
    │
    └─ generate_response()           → LLM reply using filtered products as context
```

---

## Frontend (index.html + styles.css + app.js)

Modern chat interface with:
- Dark sidebar with branding
- Chat bubbles (user right, assistant left)
- Horizontal scrollable product cards with image, name, category, price, rating, and "View Product" link
- Image upload with preview
- Conversation history (last 10 messages)
- Typing indicator
- Responsive (mobile-friendly)

---

## Dependencies (requirements.txt)

```
fastapi==0.115.0
uvicorn==0.30.0
python-dotenv==1.0.1
google-genai==1.0.0
supabase==2.10.0
pandas==2.2.0
python-multipart==0.0.9
```

---

## How to Run

1. Install dependencies: `pip install -r backend/requirements.txt`
2. Set up `.env` with API keys
3. Place real product images in `data/images/` (see product table above for filenames)
4. Run ingestion: `python backend/ingestion.py`
5. Start the server: `uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000`
6. Open `http://localhost:8000` in the browser

---

## Important Notes

- Use the Supabase MCP connection to manage the database directly
- All vector columns are exactly 3072 dimensions (Gemini Embedding 2 Preview output size)
- Handle missing product images gracefully — skip the image embedding and log a warning
- The frontend is fully functional without any build step — plain HTML/CSS/JS served by FastAPI
- The ingestion script is idempotent — running it again upserts rather than creating duplicates
