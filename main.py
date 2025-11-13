from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import numpy as np
import pickle
# import sklearn
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import Optional
import os

app = FastAPI(title="Gurgaon Real Estate Platform")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global variables for all models
pipeline = None
df = None
new_df = None
feature_text = None
locations_df = None
cosine_sim1 = None
cosine_sim2 = None
cosine_sim3 = None

def load_all_models():
    """Load all models and datasets for the three applications"""
    global pipeline, df, new_df, feature_text, locations_df, cosine_sim1, cosine_sim2, cosine_sim3
    
    # Load Price Prediction Models
    try:
        class _RemainderColsList(list):
            pass
        setattr(sklearn.compose._column_transformer, '_RemainderColsList', _RemainderColsList)
        
        with open('datasets/pipeline.pkl', 'rb') as file:
            pipeline = pickle.load(file)
        with open('datasets/df.pkl', 'rb') as file:
            df = pickle.load(file)
        print("Price prediction models loaded successfully!")
    except Exception as e:
        print(f"Error loading price prediction models: {e}")
        create_sample_prediction_data()

    # Load Visualization Data
    try:
        new_df = pd.read_csv('datasets/data_viz1.csv')
        feature_text = pickle.load(open('datasets/feature_text.pkl', 'rb'))
        print("Visualization data loaded successfully!")
    except Exception as e:
        print(f"Error loading visualization data: {e}")
        create_sample_visualization_data()

    # Load Recommendation Data
    try:
        locations_df = pickle.load(open('datasets/location_df.pkl', 'rb'))
        cosine_sim1 = pickle.load(open('datasets/cosine_sim1.pkl', 'rb'))
        cosine_sim2 = pickle.load(open('datasets/cosine_sim2.pkl', 'rb'))
        cosine_sim3 = pickle.load(open('datasets/cosine_sim3.pkl', 'rb'))
        print("Recommendation data loaded successfully!")
    except Exception as e:
        print(f"Error loading recommendation data: {e}")
        create_sample_recommendation_data()

def create_sample_prediction_data():
    """Create sample data for price prediction"""
    global df, pipeline
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

def create_sample_visualization_data():
    """Create sample data for visualization"""
    global new_df, feature_text
    np.random.seed(42)
    sectors = [f"Sector {i}" for i in [14, 15, 17, 22, 23, 29, 31, 45, 46, 50, 51, 52, 53, 54, 56, 57, 58, 65, 66, 67]]
    
    data = {
        'sector': np.random.choice(sectors, 1000),
        'property_type': np.random.choice(['flat', 'house'], 1000, p=[0.7, 0.3]),
        'bedRoom': np.random.choice([1, 2, 3, 4, 5], 1000, p=[0.1, 0.3, 0.4, 0.15, 0.05]),
        'price': np.random.uniform(5000000, 50000000, 1000),
        'price_per_sqft': np.random.uniform(5000, 20000, 1000),
        'built_up_area': np.random.uniform(500, 3000, 1000),
        'latitude': np.random.uniform(28.4, 28.6, 1000),
        'longitude': np.random.uniform(77.0, 77.2, 1000)
    }
    new_df = pd.DataFrame(data)
    feature_text = " ".join(["apartment", "house", "sector", "gurgaon", "property", "flat", "bedroom", "bathroom", 
                           "built_up_area", "price", "luxury", "modern", "spacious", "furnished"] * 100)

def create_sample_recommendation_data():
    """Create sample data for recommendation"""
    global locations_df, cosine_sim1, cosine_sim2, cosine_sim3
    sample_locations = {
        'DLF Phase 1': [0, 1500, 3000, 4500, 2000],
        'DLF Phase 2': [1500, 0, 1800, 3200, 2500],
        'DLF Phase 3': [3000, 1800, 0, 2000, 4000],
        'Sector 14': [4500, 3200, 2000, 0, 3500],
        'Sector 15': [2000, 2500, 4000, 3500, 0]
    }
    locations_df = pd.DataFrame(sample_locations, index=sample_locations.keys())
    
    n_properties = len(locations_df)
    cosine_sim1 = np.random.rand(n_properties, n_properties)
    cosine_sim2 = np.random.rand(n_properties, n_properties)
    cosine_sim3 = np.random.rand(n_properties, n_properties)
    
    cosine_sim1 = (cosine_sim1 + cosine_sim1.T) / 2
    cosine_sim2 = (cosine_sim2 + cosine_sim2.T) / 2
    cosine_sim3 = (cosine_sim3 + cosine_sim3.T) / 2

def recommend_properties_with_scores(property_name, top_n=247):
    """Recommend properties based on cosine similarity"""
    if locations_df is None or cosine_sim1 is None:
        raise ValueError("Recommendation data not loaded")
    
    cosine_sim_matrix = 30 * cosine_sim1 + 20 * cosine_sim2 + 8 * cosine_sim3
    
    try:
        property_index = locations_df.index.get_loc(property_name)
        sim_scores = list(enumerate(cosine_sim_matrix[property_index]))
    except KeyError:
        raise ValueError(f"Property '{property_name}' not found in dataset")
    
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sorted_scores[1:top_n+1]]
    top_scores = [i[1] for i in sorted_scores[1:top_n+1]]
    top_properties = locations_df.index[top_indices].tolist()
    
    recommendations_df = pd.DataFrame({
        'PropertyName': top_properties,
        'SimilarityScore': top_scores
    })
    
    return recommendations_df

def matplotlib_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    buf.close()
    return img_str

# Load all models on startup
load_all_models()

# ==================== ROUTES ====================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with all three applications"""
    return templates.TemplateResponse("index.html", {"request": request})

# ==================== PRICE PREDICTION ROUTES ====================

@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    """Price prediction form"""
    if df is None:
        raise HTTPException(status_code=500, detail="Price prediction data not loaded")
    
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

@app.post("/predict-price")
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
    """Predict property price"""
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Price prediction model not loaded")
    
    try:
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
        
        base_price = np.expm1(pipeline.predict(input_data))[0]
        low = base_price - 0.22
        high = base_price + 0.22
        
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

# ==================== VISUALIZATION ROUTES ====================

@app.get("/visualize", response_class=HTMLResponse)
async def visualize_page(request: Request):
    """Visualization dashboard"""
    if new_df is None:
        raise HTTPException(status_code=500, detail="Visualization data not loaded")
    
    # Prepare initial visualizations
    group_df = new_df.groupby('sector')[['price','price_per_sqft','built_up_area','latitude','longitude']].mean()
    
    map_plot = create_sector_price_map(group_df)
    wordcloud_img = create_wordcloud(feature_text)
    scatter_plot_flat = create_scatter_plot(new_df, 'flat')
    scatter_plot_house = create_scatter_plot(new_df, 'house')
    pie_chart_overall = create_pie_chart(new_df, 'Overall')
    box_plot = create_box_plot(new_df)
    dist_plot = create_dist_plot(new_df)
    
    sectors = ['Overall'] + sorted(new_df['sector'].unique().tolist())
    
    return templates.TemplateResponse("visualize.html", {
        "request": request,
        "map_plot": map_plot,
        "wordcloud_img": wordcloud_img,
        "scatter_plot_flat": scatter_plot_flat,
        "scatter_plot_house": scatter_plot_house,
        "pie_chart_overall": pie_chart_overall,
        "box_plot": box_plot,
        "dist_plot": dist_plot,
        "sectors": sectors
    })

@app.post("/update-scatter-plot")
async def update_scatter_plot(property_type: str = Form(...)):
    """Update scatter plot based on property type"""
    scatter_plot = create_scatter_plot(new_df, property_type)
    return JSONResponse({"plot_html": scatter_plot})

@app.post("/update-pie-chart")
async def update_pie_chart(sector: str = Form(...)):
    """Update pie chart based on sector selection"""
    pie_chart = create_pie_chart(new_df, sector)
    return JSONResponse({"plot_html": pie_chart})

def create_sector_price_map(group_df):
    """Create sector price per sqft map"""
    fig = px.scatter_mapbox(
        group_df, 
        lat="latitude", 
        lon="longitude", 
        color="price_per_sqft", 
        size='built_up_area',
        color_continuous_scale=px.colors.cyclical.IceFire, 
        zoom=10,
        mapbox_style="open-street-map",
        width=1200,
        height=700,
        hover_name=group_df.index,
        title="Sector Price per Sqft Map"
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def create_wordcloud(feature_text):
    """Create word cloud"""
    plt.rcParams["font.family"] = "Arial"
    wordcloud = WordCloud(
        width=800, 
        height=800, 
        background_color='white', 
        stopwords=set(['s']),
        min_font_size=10
    ).generate(feature_text)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.set_title("Word Cloud of Property Features", fontsize=16, pad=20)
    
    img_base64 = matplotlib_to_base64(fig)
    plt.close(fig)
    return img_base64

def create_scatter_plot(df, property_type):
    """Create built-up area vs price scatter plot"""
    if property_type == 'house':
        filtered_df = df[df['property_type'] == 'house']
        title = "House: Built-up Area vs Price"
    else:
        filtered_df = df[df['property_type'] == 'flat']
        title = "Flat: Built-up Area vs Price"
    
    fig = px.scatter(
        filtered_df, 
        x='built_up_area', 
        y='price', 
        color='bedRoom',
        title=title,
        labels={
            'built_up_area': 'Built-up Area (sq ft)',
            'price': 'Price (₹)',
            'bedRoom': 'Bedrooms'
        }
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)

def create_pie_chart(df, sector):
    """Create BHK distribution pie chart"""
    if sector == 'Overall':
        filtered_df = df
        title = "Overall BHK Distribution"
    else:
        filtered_df = df[df['sector'] == sector]
        title = f"BHK Distribution in {sector}"
    
    bed_count = filtered_df['bedRoom'].value_counts().sort_index()
    labels = [f'{int(bed)} BHK' for bed in bed_count.index]
    
    fig = px.pie(
        values=bed_count.values,
        names=labels,
        title=title,
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)

def create_box_plot(df):
    """Create BHK price comparison box plot"""
    fig = px.box(
        df, 
        x='bedRoom', 
        y='price',
        title="BHK Price Comparison",
        labels={
            'bedRoom': 'Number of Bedrooms',
            'price': 'Price (₹)'
        }
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)

def create_dist_plot(df):
    """Create distribution plot of property types"""
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.histplot(df[df['property_type'] == 'flat']['price'], 
                 label='Flat', color='blue', kde=True, ax=ax, alpha=0.7)
    sns.histplot(df[df['property_type'] == 'house']['price'], 
                 label='House', color='orange', kde=True, ax=ax, alpha=0.7)
    
    ax.set_title('Price Distribution by Property Type', fontsize=16, fontweight='bold')
    ax.set_xlabel('Price (₹)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    img_base64 = matplotlib_to_base64(fig)
    plt.close(fig)
    return img_base64

# ==================== RECOMMENDATION ROUTES ====================

@app.get("/recommend", response_class=HTMLResponse)
async def recommend_page(request: Request):
    """Property recommendation page"""
    if locations_df is None:
        raise HTTPException(status_code=500, detail="Recommendation data not loaded")
    
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
        raise HTTPException(status_code=500, detail="Recommendation data not loaded")
    
    try:
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
    """Get property recommendations"""
    if locations_df is None:
        raise HTTPException(status_code=500, detail="Recommendation data not loaded")
    
    try:
        recommendation_df = recommend_properties_with_scores(selected_apartment)
        recommendation_df = recommendation_df.sort_values(by='SimilarityScore', ascending=False).reset_index(drop=True).head(10)
        
        recommendations = recommendation_df.to_dict('records')
        
        return JSONResponse({
            "recommendations": recommendations,
            "selected_apartment": selected_apartment,
            "count": len(recommendations)
        })
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Recommendation error: {str(e)}")

# ==================== API ENDPOINTS ====================

@app.get("/api/options/prediction")
async def get_prediction_options():
    """Get all options for prediction dropdowns"""
    if df is None:
        raise HTTPException(status_code=500, detail="Prediction data not loaded")
    
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

@app.get("/api/options/recommendation")
async def get_recommendation_options():
    """Get all options for recommendation dropdowns"""
    if locations_df is None:
        raise HTTPException(status_code=500, detail="Recommendation data not loaded")
    
    return JSONResponse({
        "locations": sorted(locations_df.columns.to_list()),
        "apartments": sorted(locations_df.index.to_list())
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)