from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import numpy as np
import pickle
from typing import List, Optional

app = FastAPI(title="Gurgaon Property Recommender")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global variables for data
locations_df = None
cosine_sim1 = None
cosine_sim2 = None
cosine_sim3 = None

def load_data():
    """Load the recommendation data with error handling"""
    global locations_df, cosine_sim1, cosine_sim2, cosine_sim3
    
    try:
        # Load your data files (update paths as needed)
        locations_df = pickle.load(open('datasets/location_df.pkl', 'rb'))
        cosine_sim1 = pickle.load(open('datasets/cosine_sim1.pkl', 'rb'))
        cosine_sim2 = pickle.load(open('datasets/cosine_sim2.pkl', 'rb'))
        cosine_sim3 = pickle.load(open('datasets/cosine_sim3.pkl', 'rb'))
        
        print("Recommendation data loaded successfully!")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating sample data for demonstration...")
        create_sample_data()

def create_sample_data():
    """Create sample data if files are not available"""
    global locations_df, cosine_sim1, cosine_sim2, cosine_sim3
    
    # Sample locations data
    sample_locations = {
        'DLF Phase 1': [0, 1500, 3000, 4500, 2000],
        'DLF Phase 2': [1500, 0, 1800, 3200, 2500],
        'DLF Phase 3': [3000, 1800, 0, 2000, 4000],
        'Sector 14': [4500, 3200, 2000, 0, 3500],
        'Sector 15': [2000, 2500, 4000, 3500, 0]
    }
    
    locations_df = pd.DataFrame(sample_locations, index=sample_locations.keys())
    
    # Sample similarity matrices
    n_properties = len(locations_df)
    cosine_sim1 = np.random.rand(n_properties, n_properties)
    cosine_sim2 = np.random.rand(n_properties, n_properties)
    cosine_sim3 = np.random.rand(n_properties, n_properties)
    
    # Make matrices symmetric
    cosine_sim1 = (cosine_sim1 + cosine_sim1.T) / 2
    cosine_sim2 = (cosine_sim2 + cosine_sim2.T) / 2
    cosine_sim3 = (cosine_sim3 + cosine_sim3.T) / 2

def recommend_properties_with_scores(property_name, top_n=247):
    """Recommend properties based on cosine similarity (same as Streamlit)"""
    if locations_df is None or cosine_sim1 is None:
        raise ValueError("Data not loaded")
    
    # Same weighted combination as Streamlit
    cosine_sim_matrix = 30 * cosine_sim1 + 20 * cosine_sim2 + 8 * cosine_sim3
    
    # Get the similarity scores for the property using its name as the index
    try:
        property_index = locations_df.index.get_loc(property_name)
        sim_scores = list(enumerate(cosine_sim_matrix[property_index]))
    except KeyError:
        raise ValueError(f"Property '{property_name}' not found in dataset")
    
    # Sort properties based on the similarity scores
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the indices and scores of the top_n most similar properties
    top_indices = [i[0] for i in sorted_scores[1:top_n+1]]
    top_scores = [i[1] for i in sorted_scores[1:top_n+1]]
    
    # Retrieve the names of the top properties using the indices
    top_properties = locations_df.index[top_indices].tolist()
    
    # Create a dataframe with the results
    recommendations_df = pd.DataFrame({
        'PropertyName': top_properties,
        'SimilarityScore': top_scores
    })
    
    return recommendations_df

# Load data on startup
load_data()

@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    """Serve the main recommendation page"""
    if locations_df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    locations_list = sorted(locations_df.columns.to_list())
    apartments_list = sorted(locations_df.index.to_list())
    
    return templates.TemplateResponse("recommend.html", {
        "request": request,
        "locations": locations_list,
        "apartments": apartments_list
    })

@app.post("/search-properties")
async def search_properties(
    selected_location: str = Form(...),
    radius: float = Form(...)
):
    """Search properties within given radius"""
    if locations_df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    try:
        # Same logic as Streamlit
        result_ser = locations_df[locations_df[selected_location] < radius * 1000][selected_location].sort_values()
        
        apartments = []
        for key, value in result_ser.items():
            apartments.append({
                "name": str(key),
                "distance": round(value/1000, 2),
                "display": f"{key} ----------> {round(value/1000, 2)} kms"
            })
        
        return JSONResponse({
            "apartments": apartments,
            "count": len(apartments)
        })
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Search error: {str(e)}")

@app.post("/get-recommendations")
async def get_recommendations(selected_apartment: str = Form(...)):
    """Get property recommendations based on selected apartment"""
    if locations_df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    try:
        # Get recommendations (same as Streamlit)
        recommendation_df = recommend_properties_with_scores(selected_apartment)
        recommendation_df = recommendation_df.sort_values(by='SimilarityScore', ascending=False).reset_index(drop=True).head(10)
        
        # Convert to dictionary for JSON response
        recommendations = recommendation_df.to_dict('records')
        
        return JSONResponse({
            "recommendations": recommendations,
            "selected_apartment": selected_apartment,
            "count": len(recommendations)
        })
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Recommendation error: {str(e)}")

@app.get("/api/locations")
async def get_locations():
    """API endpoint to get all available locations"""
    if locations_df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    return JSONResponse({
        "locations": sorted(locations_df.columns.to_list()),
        "apartments": sorted(locations_df.index.to_list())
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)