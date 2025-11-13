// Common utility functions
function showLoading(elementId) {
    document.getElementById(elementId).innerHTML = `
        <div class="text-center py-4">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-2 text-muted">Loading...</p>
        </div>
    `;
}

function showError(elementId, message) {
    document.getElementById(elementId).innerHTML = `
        <div class="alert alert-danger">
            <i class="fas fa-exclamation-triangle"></i> ${message}
        </div>
    `;
}

function formatPrice(price) {
    if (price >= 10000000) {
        return '₹' + (price / 10000000).toFixed(2) + ' Cr';
    } else {
        return '₹' + (price / 100000).toFixed(2) + ' L';
    }
}

// Price Prediction Form Handler
document.addEventListener('DOMContentLoaded', function() {
    // Initialize all forms if they exist on the page
    
    // Price Prediction Form
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        initializePredictionForm(predictionForm);
    }
    
    // Recommendation Forms
    const searchForm = document.getElementById('searchForm');
    if (searchForm) {
        initializeSearchForm(searchForm);
    }
    
    const recommendationForm = document.getElementById('recommendationForm');
    if (recommendationForm) {
        initializeRecommendationForm(recommendationForm);
    }
});

function initializePredictionForm(form) {
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        if (!form.checkValidity()) {
            e.stopPropagation();
            form.classList.add('was-validated');
            return;
        }
        
        const resultDiv = document.getElementById('predictionResult');
        const loadingSpinner = document.getElementById('loadingSpinner');
        const submitButton = form.querySelector('button[type="submit"]');
        
        // Show loading
        loadingSpinner.style.display = 'block';
        resultDiv.style.display = 'none';
        submitButton.disabled = true;
        
        try {
            const formData = new FormData(form);
            const response = await fetch('/predict-price', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (response.ok) {
                displayPredictionResult(result);
            } else {
                throw new Error(result.detail || 'Prediction failed');
            }
        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            loadingSpinner.style.display = 'none';
            submitButton.disabled = false;
        }
    });
    
    // Add real-time validation
    const inputs = form.querySelectorAll('select, input');
    inputs.forEach(input => {
        input.addEventListener('change', function() {
            if (this.checkValidity()) {
                this.classList.remove('is-invalid');
                this.classList.add('is-valid');
            } else {
                this.classList.remove('is-valid');
                this.classList.add('is-invalid');
            }
        });
    });
}

function displayPredictionResult(result) {
    const resultDiv = document.getElementById('predictionResult');
    const resultText = document.getElementById('resultText');
    const priceRange = document.getElementById('priceRange');
    
    resultText.textContent = result.prediction;
    priceRange.textContent = result.price_range;
    
    resultDiv.style.display = 'block';
    resultDiv.scrollIntoView({ behavior: 'smooth' });
}

function initializeSearchForm(form) {
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData(form);
        const searchLoading = document.getElementById('searchLoading');
        const searchResults = document.getElementById('searchResults');
        
        // Show loading
        searchLoading.style.display = 'block';
        searchResults.style.display = 'none';
        
        try {
            const response = await fetch('/search-properties', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (response.ok) {
                displaySearchResults(data);
            } else {
                throw new Error(data.detail || 'Search failed');
            }
        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            searchLoading.style.display = 'none';
        }
    });
}

function displaySearchResults(data) {
    const searchResults = document.getElementById('searchResults');
    const resultsCount = document.getElementById('resultsCount');
    const apartmentsList = document.getElementById('apartmentsList');
    const recommendationSelection = document.getElementById('recommendationSelection');
    const apartmentsRadioGroup = document.getElementById('apartmentsRadioGroup');
    
    resultsCount.textContent = data.count;
    
    if (data.apartments.length === 0) {
        apartmentsList.innerHTML = '<div class="alert alert-warning">No properties found within the specified radius.</div>';
        recommendationSelection.style.display = 'none';
    } else {
        let apartmentsHTML = '';
        data.apartments.forEach(apt => {
            apartmentsHTML += `
                <div class="property-card">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <h6 class="mb-1">${apt.name}</h6>
                            <p class="text-muted mb-0">Distance from selected location</p>
                        </div>
                        <div class="distance-badge">${apt.distance} kms</div>
                    </div>
                </div>
            `;
        });
        apartmentsList.innerHTML = apartmentsHTML;
        
        let radioHTML = '';
        data.apartments.forEach(apt => {
            radioHTML += `
                <div class="apartment-option" onclick="selectApartment('${apt.name}')">
                    <div class="form-check">
                        <input class="form-check-input" type="radio" name="selectedApartment" 
                               id="apt_${apt.name}" value="${apt.name}">
                        <label class="form-check-label" for="apt_${apt.name}">
                            <strong>${apt.name}</strong> - ${apt.distance} kms away
                        </label>
                    </div>
                </div>
            `;
        });
        apartmentsRadioGroup.innerHTML = radioHTML;
        recommendationSelection.style.display = 'block';
    }
    
    searchResults.style.display = 'block';
}

function initializeRecommendationForm(form) {
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData(form);
        const recommendationLoading = document.getElementById('recommendationLoading');
        const recommendationResults = document.getElementById('recommendationResults');
        
        // Show loading
        recommendationLoading.style.display = 'block';
        recommendationResults.style.display = 'none';
        
        try {
            const response = await fetch('/get-recommendations', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (response.ok) {
                displayRecommendationResults(data);
            } else {
                throw new Error(data.detail || 'Recommendation failed');
            }
        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            recommendationLoading.style.display = 'none';
        }
    });
}

function displayRecommendationResults(data) {
    const recommendationResults = document.getElementById('recommendationResults');
    const selectedApartmentName = document.getElementById('selectedApartmentName');
    const recommendationsList = document.getElementById('recommendationsList');
    
    selectedApartmentName.textContent = data.selected_apartment;
    
    if (data.recommendations.length === 0) {
        recommendationsList.innerHTML = '<div class="alert alert-warning">No recommendations found.</div>';
    } else {
        let recommendationsHTML = '';
        data.recommendations.forEach((rec, index) => {
            const similarityPercent = (rec.SimilarityScore * 100).toFixed(1);
            recommendationsHTML += `
                <div class="property-card">
                    <div class="d-flex justify-content-between align-items-start">
                        <div class="flex-grow-1">
                            <div class="d-flex align-items-center mb-2">
                                <span class="badge bg-primary me-2">#${index + 1}</span>
                                <h6 class="mb-0">${rec.PropertyName}</h6>
                            </div>
                            <div class="d-flex align-items-center">
                                <span class="similarity-badge me-2">
                                    <i class="fas fa-chart-line me-1"></i>${similarityPercent}% Match
                                </span>
                                <small class="text-muted">Similarity Score: ${rec.SimilarityScore.toFixed(4)}</small>
                            </div>
                        </div>
                        <button class="btn btn-outline-primary btn-sm" onclick="selectThisApartment('${rec.PropertyName}')">
                            <i class="fas fa-redo"></i>
                        </button>
                    </div>
                </div>
            `;
        });
        recommendationsList.innerHTML = recommendationsHTML;
    }
    
    recommendationResults.style.display = 'block';
}

// Global functions for apartment selection
function selectApartment(apartmentName) {
    window.selectedApartmentForRecommendation = apartmentName;
    
    document.querySelectorAll('.apartment-option').forEach(option => {
        option.classList.remove('selected');
    });
    event.currentTarget.classList.add('selected');
    
    document.getElementById('selected_apartment').value = apartmentName;
    
    document.getElementById('recommendationForm').scrollIntoView({ 
        behavior: 'smooth',
        block: 'start'
    });
}

function selectThisApartment(apartmentName) {
    document.getElementById('selected_apartment').value = apartmentName;
    document.getElementById('recommendationForm').scrollIntoView({ 
        behavior: 'smooth',
        block: 'center'
    });
}