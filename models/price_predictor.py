from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import numpy as np
import pickle
import sklearn
from typing import Optional

app = FastAPI(title="Gurgaon Property Price Predictor")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global variables for model and data
pipeline = None
df = None

def load_model():
    """Load the ML model and data with error handling"""
    global pipeline, df
    
    try:
        # Try Solution 4 first (same as your Streamlit code)
        class _RemainderColsList(list):
            pass
        setattr(sklearn.compose._column_transformer, '_RemainderColsList', _RemainderColsList)
        
        with open('pipeline.pkl', 'rb') as file:
            pipeline = pickle.load(file)
        with open('df.pkl', 'rb') as file:
            df = pickle.load(file)
            
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please re-train the model with current scikit-learn version")
        
        # Create sample data for demonstration
        create_sample_data()

def create_sample_data():
    """Create sample data if model files are not available"""
    global df, pipeline
    
    # Sample DataFrame structure
    sample_data = {
        'property_type': ['flat', 'house', 'flat', 'house'],
        'sector': ['Sector 14', 'Sector 15', 'Sector 17', 'Sector 22'],
        'bedRoom': [2.0, 3.0, 2.0, 4.0],
        'bathroom': [2.0, 3.0, 2.0, 3.0],
        'balcony': [1, 2, 1, 3],
        'agePossession': ['New Property', '1-5 years', '6-10 years', '10+ years'],
        'built_up_area': [1200.0, 1800.0, 1100.0, 2200.0],
        'servant room': [0.0, 1.0, 0.0, 1.0],
        'store room': [0.0, 1.0, 0.0, 1.0],
        'furnishing_type': ['Semi-Furnished', 'Fully-Furnished', 'Unfurnished', 'Fully-Furnished'],
        'luxury_category': ['Low', 'Medium', 'Low', 'High'],
        'floor_category': ['Low Rise', 'Mid Rise', 'Low Rise', 'High Rise']
    }
    
    df = pd.DataFrame(sample_data)
    pipeline = None

# Load model on startup
load_model()

@app.get("/", response_class=HTMLResponse)
async def predict_form(request: Request):
    """Serve the prediction form with dynamic dropdown options"""
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    # Get unique values for dropdowns
    property_types = df['property_type'].unique().tolist()
    sectors = sorted(df['sector'].unique().tolist())
    bedrooms = sorted(df['bedRoom'].unique().tolist())
    bathrooms = sorted(df['bathroom'].unique().tolist())
    balconies = sorted(df['balcony'].unique().tolist())
    age_possessions = sorted(df['agePossession'].unique().tolist())
    servant_rooms = sorted(df['servant room'].unique().tolist())
    store_rooms = sorted(df['store room'].unique().tolist())
    furnishing_types = df['furnishing_type'].unique().tolist()
    luxury_categories = df['luxury_category'].unique().tolist()
    floor_categories = df['floor_category'].unique().tolist()
    
    return templates.TemplateResponse("predict.html", {
        "request": request,
        "property_types": property_types,
        "sectors": sectors,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "balconies": balconies,
        "age_possessions": age_possessions,
        "servant_rooms": servant_rooms,
        "store_rooms": store_rooms,
        "furnishing_types": furnishing_types,
        "luxury_categories": luxury_categories,
        "floor_categories": floor_categories
    })

@app.post("/predict")
async def predict_price(
    property_type: str = Form(...),
    sector: str = Form(...),
    bedRoom: float = Form(...),
    bathroom: float = Form(...),
    balcony: int = Form(...),
    agePossession: str = Form(...),
    built_up_area: float = Form(...),
    servant_room: float = Form(...),
    store_room: float = Form(...),
    furnishing_type: str = Form(...),
    luxury_category: str = Form(...),
    floor_category: str = Form(...)
):
    """Predict property price based on input features"""
    
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please re-train the model with current scikit-learn version")
    
    try:
        # Create input DataFrame (same as Streamlit)
        input_data = pd.DataFrame({
            'property_type': [property_type],
            'sector': [sector],
            'bedRoom': [bedRoom],
            'bathroom': [bathroom],
            'balcony': [balcony],
            'agePossession': [agePossession],
            'built_up_area': [built_up_area],
            'servant room': [servant_room],
            'store room': [store_room],
            'furnishing_type': [furnishing_type],
            'luxury_category': [luxury_category],
            'floor_category': [floor_category]
        })
        
        # Predict (same logic as Streamlit)
        base_price = np.expm1(pipeline.predict(input_data))[0]
        low = base_price - 0.22
        high = base_price + 0.22
        
        # Format response
        result = {
            "prediction": f"The price of the flat is between {round(low, 2)} cr and {round(high, 2)} cr",
            "low_price": round(low, 2),
            "high_price": round(high, 2),
            "base_price": round(base_price, 2),
            "price_range": f"{round(low, 2)} cr - {round(high, 2)} cr"
        }
        
        return JSONResponse(result)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/api/options")
async def get_options():
    """API endpoint to get all available options for dropdowns"""
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    options = {
        "property_types": df['property_type'].unique().tolist(),
        "sectors": sorted(df['sector'].unique().tolist()),
        "bedrooms": sorted(df['bedRoom'].unique().tolist()),
        "bathrooms": sorted(df['bathroom'].unique().tolist()),
        "balconies": sorted(df['balcony'].unique().tolist()),
        "age_possessions": sorted(df['agePossession'].unique().tolist()),
        "servant_rooms": sorted(df['servant room'].unique().tolist()),
        "store_rooms": sorted(df['store room'].unique().tolist()),
        "furnishing_types": df['furnishing_type'].unique().tolist(),
        "luxury_categories": df['luxury_category'].unique().tolist(),
        "floor_categories": df['floor_category'].unique().tolist()
    }
    
    return JSONResponse(options)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)