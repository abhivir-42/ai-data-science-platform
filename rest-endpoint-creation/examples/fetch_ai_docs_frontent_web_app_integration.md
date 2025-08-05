Build a Frontend Web Application with uAgents and Open Food Facts API

This guide demonstrates how to create a complete web application that integrates uAgents with external APIs using a Flask frontend. We'll build a food product discovery system using the Open Food Facts API.

Overview

The Frontend Integration example provides:

Two Specialized uAgents - Search Agent and Info Agent
REST API Endpoints using uAgents framework
External API Integration with Open Food Facts
Modern Web Interface built with Flask and HTML/CSS/JavaScript
Real-time Health Monitoring of all services
Prerequisites

Before you begin, ensure you have:

Python 3.11+ installed
Basic knowledge of Flask and web development
Understanding of REST APIs and HTTP requests
Installation

1. Clone the Complete Example

git clone https://github.com/fetchai/innovation-lab-examples.git
cd innovation-lab-examples/frontend-integration

2. Set Up Virtual Environment

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

3. Install Dependencies

pip install -r requirements.txt

Architecture Overview

Two Specialized uAgents

Our system consists of two specialized microservices:

Search Agent (Port 8001): Handles product search queries
Info Agent (Port 8002): Retrieves detailed product information
Frontend Application

Flask Web App (Port 5000): Modern web interface to interact with agents
Quick Start

1. Start Search Agent

Terminal 1:

source venv/bin/activate  # Activate venv if not already active
python3 product_search_agent.py

2. Start Info Agent

Terminal 2:

source venv/bin/activate  # Activate venv if not already active
python3 product_info_agent.py

3. Start Frontend

Terminal 3:

source venv/bin/activate  # Activate venv if not already active
python3 frontend_app.py

4. Access the Web Interface

Open your browser to: http://127.0.0.1:5000

Frontend Browser

Agent Implementation Details

Search Agent (Port 8001)

The search agent handles product discovery using natural language queries:

from uagents import Agent, Context, Model
import openfoodfacts

class SearchRequest(Model):
    query: str

class ProductInfo(Model):
    code: str
    product_name: str
    brands: str
    categories: str
    image_url: str

class SearchResponse(Model):
    query: str
    count: int
    products: List[ProductInfo]
    error: Optional[str] = None

search_agent = Agent(
    name="product_search_agent",
    port=8001,
    endpoint=["http://127.0.0.1:8001/submit"],
)

@search_agent.on_rest_post("/search", SearchRequest, SearchResponse)
async def search_products(ctx: Context, req: SearchRequest) -> SearchResponse:
    try:
        query = req.query
        ctx.logger.info(f"Searching for products with query: {query}")
        
        # Search products using Open Food Facts API
        results = api.product.text_search(query, page_size=10)
        
        # Extract relevant information
        products = []
        for product in results.get("products", [])[:10]:
            product_info = ProductInfo(
                code=product.get("code", "N/A"),
                product_name=product.get("product_name", "N/A"),
                brands=product.get("brands", "N/A"),
                categories=product.get("categories", "N/A"),
                image_url=product.get("image_url", "")
            )
            products.append(product_info)
        
        return SearchResponse(
            query=query,
            count=results.get("count", 0),
            products=products
        )
    except Exception as e:
        return SearchResponse(
            query=req.query,
            count=0,
            products=[],
            error=f"Failed to search products: {str(e)}"
        )

Info Agent (Port 8002)

The info agent provides detailed product information using exact barcodes:

from uagents import Agent, Context, Model
import requests

class ProductRequest(Model):
    barcode: str

class ProductInfoResponse(Model):
    barcode: str
    product_name: str
    brands: str
    categories: str
    ingredients_text: str
    allergens: str
    nutrition_grades: str
    ecoscore_grade: str
    image_url: str
    countries: str
    stores: str
    packaging: str
    quantity: str
    energy_100g: str
    fat_100g: str
    sugars_100g: str
    salt_100g: str
    error: Optional[str] = None

info_agent = Agent(
    name="product_info_agent",
    port=8002,
    endpoint=["http://127.0.0.1:8002/submit"],
)

@info_agent.on_rest_post("/product", ProductRequest, ProductInfoResponse)
async def get_product_info(ctx: Context, req: ProductRequest) -> ProductInfoResponse:
    try:
        barcode = req.barcode
        ctx.logger.info(f"Getting product info for barcode: {barcode}")
        
        # Use direct API call to Open Food Facts
        url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
        headers = {
            'User-Agent': 'uAgents-FoodInfo/1.0 (https://github.com/fetchai/uAgents)'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        
        if data.get('status') != 1 or "product" not in data:
            return ProductInfoResponse(
                barcode=barcode,
                # ... other fields with "N/A"
                error="Product not found"
            )
        
        product = data["product"]
        
        # Extract comprehensive product information
        return ProductInfoResponse(
            barcode=barcode,
            product_name=product.get("product_name", "N/A"),
            brands=product.get("brands", "N/A"),
            categories=product.get("categories", "N/A"),
            ingredients_text=product.get("ingredients_text", "N/A"),
            allergens=product.get("allergens", "N/A"),
            nutrition_grades=product.get("nutrition_grades", "N/A"),
            ecoscore_grade=product.get("ecoscore_grade", "N/A"),
            image_url=product.get("image_url", ""),
            countries=product.get("countries", "N/A"),
            stores=product.get("stores", "N/A"),
            packaging=product.get("packaging", "N/A"),
            quantity=product.get("quantity", "N/A"),
            energy_100g=str(product.get("nutriments", {}).get("energy_100g", "N/A")),
            fat_100g=str(product.get("nutriments", {}).get("fat_100g", "N/A")),
            sugars_100g=str(product.get("nutriments", {}).get("sugars_100g", "N/A")),
            salt_100g=str(product.get("nutriments", {}).get("salt_100g", "N/A"))
        )
    except Exception as e:
        return ProductInfoResponse(
            barcode=req.barcode,
            # ... other fields with "N/A"
            error=f"Failed to get product info: {str(e)}"
        )

Frontend Implementation

Flask Backend

The Flask application serves as a bridge between the web interface and uAgents:

from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# Agent endpoints
AGENTS = {
    "search": "http://127.0.0.1:8001",
    "info": "http://127.0.0.1:8002"
}

@app.route('/search_products', methods=['POST'])
def search_products():
    """Search products via search agent"""
    try:
        query = request.form.get('query', '').strip()
        if not query:
            return jsonify({"error": "Please provide a search query"})
        
        # Call search agent with POST request
        payload = {"query": query}
        response = requests.post(f"{AGENTS['search']}/search", json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        # Format the results for display
        formatted_results = []
        if result.get('products'):
            for product in result['products'][:10]:
                formatted_product = {
                    'name': product.get('product_name', 'N/A'),
                    'brands': product.get('brands', 'N/A'),
                    'barcode': product.get('code', 'N/A'),
                    'categories': product.get('categories', 'N/A'),
                    'image_url': product.get('image_url', '')
                }
                formatted_results.append(formatted_product)
        
        return jsonify({
            "success": True, 
            "count": result.get('count', 0),
            "query": query,
            "products": formatted_results
        })
        
    except requests.RequestException as e:
        return jsonify({"error": f"Failed to connect to search agent: {str(e)}"})
    except Exception as e:
        return jsonify({"error": f"Search failed: {str(e)}"})

@app.route('/health')
def health_check():
    """Check health of all agents"""
    health_status = {}
    
    for agent_name, agent_url in AGENTS.items():
        try:
            response = requests.get(f"{agent_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                health_status[agent_name] = {
                    "status": "healthy", 
                    "url": agent_url, 
                    "agent_info": health_data
                }
            else:
                health_status[agent_name] = {"status": "unhealthy", "url": agent_url}
        except:
            health_status[agent_name] = {"status": "offline", "url": agent_url}
    
    return jsonify(health_status)

Testing the Application

1. Product Search Testing

Search Query: "chocolate"

Navigate to http://127.0.0.1:5000
In the "Search Products" section, enter chocolate in the search field
Click "Search Products" to see results
Search Results for Chocolate

The search will return multiple chocolate products with product names, brands, categories, barcodes, and images.

2. Product Details Testing

Barcode Query: "3017624010701"

In the "Get Product Information" section, enter the barcode 3017624010701
Click "Get Product Info" to retrieve detailed information
Product Details for Nutella

This will display comprehensive information for Nutella including basic information, ingredients, nutrition facts, and quality scores.

3. Health Status Monitoring

Click the "Check Agent Status" button to verify all services are running:

Agent Health Status

This displays the real-time status of both Search Agent (Port 8001) and Info Agent (Port 8002).

Key Features Demonstrated

1. Microservice Architecture

Separation of Concerns: Each agent has a specific responsibility
Independent Scaling: Agents can be scaled independently
Fault Isolation: Failure in one service doesn't affect others
2. REST API Design

Proper HTTP Methods: POST for data operations, GET for health checks
Pydantic Models: Type-safe request/response validation
Error Handling: Comprehensive error responses
3. Frontend Integration

Ajax Requests: Asynchronous communication with agents
Real-time Updates: Dynamic UI updates without page refresh
Health Monitoring: Live status checking of backend services
4. External API Integration

HTTP Client Usage: Direct API calls to Open Food Facts
Data Transformation: Converting external API responses to internal models
Error Handling: Graceful handling of external API failures
GitHub Repository

For the complete code and additional examples, visit the Frontend Integration Example repository.

This repository includes:

✅ Complete agent implementations
✅ Flask web application
✅ Modern responsive web interface
✅ Docker configuration
✅ Comprehensive documentation
✅ Testing examples