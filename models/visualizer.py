from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import io
import base64
from typing import Optional

app = FastAPI(title="Gurgaon Property Visualization")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load your data (update paths as needed)
try:
    new_df = pd.read_csv('datasets/data_viz1.csv')
    feature_text = pickle.load(open('datasets/feature_text.pkl', 'rb'))
except FileNotFoundError:
    # Fallback: generate sample data if files not found
    new_df = generate_sample_data()
    feature_text = " ".join(["apartment", "house", "sector", "gurgaon", "property", "flat", "bedroom", "bathroom", 
                           "built_up_area", "price", "luxury", "modern", "spacious", "furnished"] * 100)

def generate_sample_data():
    """Generate sample data if original files are not available"""
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
    return pd.DataFrame(data)

def matplotlib_to_base64(fig):
    """Convert matplotlib figure to base64 string for HTML embedding"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    buf.close()
    return img_str

@app.get("/", response_class=HTMLResponse)
async def visualization_dashboard(request: Request):
    """Main visualization dashboard"""
    
    # Prepare data for the template
    group_df = new_df.groupby('sector')[['price','price_per_sqft','built_up_area','latitude','longitude']].mean()
    
    # Convert plots to HTML components
    map_plot = create_sector_price_map(group_df)
    wordcloud_img = create_wordcloud(feature_text)
    scatter_plot_flat = create_scatter_plot(new_df, 'flat')
    scatter_plot_house = create_scatter_plot(new_df, 'house')
    pie_chart_overall = create_pie_chart(new_df, 'Overall')
    box_plot = create_box_plot(new_df)
    dist_plot = create_dist_plot(new_df)
    
    # Get unique sectors for dropdown
    sectors = ['Overall'] + sorted(new_df['sector'].unique().tolist())
    
    return templates.TemplateResponse("visualization.html", {
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
    """Update scatter plot based on property type selection"""
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
    """Create word cloud and convert to base64"""
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
    
    fig.update_layout(
        xaxis_title="Built-up Area (sq ft)",
        yaxis_title="Price (₹)",
        showlegend=True
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
    
    # Count bedrooms and create labels
    bed_count = filtered_df['bedRoom'].value_counts().sort_index()
    labels = [f'{int(bed)} BHK' for bed in bed_count.index]
    
    fig = px.pie(
        values=bed_count.values,
        names=labels,
        title=title,
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}"
    )
    
    return fig.to_html(full_html=False, include_plotlyjs=False)

def create_box_plot(df):
    """Create side by side BHK price comparison box plot"""
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
    
    fig.update_layout(
        xaxis_title="Number of Bedrooms",
        yaxis_title="Price (₹)",
        showlegend=False
    )
    
    return fig.to_html(full_html=False, include_plotlyjs=False)

def create_dist_plot(df):
    """Create distribution plot of property types and convert to base64"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create distribution plots
    sns.histplot(df[df['property_type'] == 'flat']['price'], 
                 label='Flat', color='blue', kde=True, ax=ax, alpha=0.7)
    sns.histplot(df[df['property_type'] == 'house']['price'], 
                 label='House', color='orange', kde=True, ax=ax, alpha=0.7)
    
    # Customize the plot
    ax.set_title('Price Distribution by Property Type', fontsize=16, fontweight='bold')
    ax.set_xlabel('Price (₹)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Convert to base64
    img_base64 = matplotlib_to_base64(fig)
    plt.close(fig)
    
    return img_base64

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)