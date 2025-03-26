document.addEventListener('DOMContentLoaded', function() {
    // Diabetes prediction form handling
    const diabetesForm = document.getElementById('diabetesForm');
    const diabetesResult = document.getElementById('diabetesResult');
    let diabetesChart = null;
    let diabetesFeatureChart = null;
    
    if (diabetesForm) {
        diabetesForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(diabetesForm);
            const features = [
                parseFloat(formData.get('glucose')),
                parseFloat(formData.get('bloodPressure')),
                parseFloat(formData.get('bmi')),
                parseFloat(formData.get('age')),
                parseFloat(formData.get('insulin')),
                parseFloat(formData.get('skinThickness')),
                parseFloat(formData.get('pregnancies')),
                parseFloat(formData.get('dpf'))
            ];
            
            try {
                const response = await fetch('/predict/diabetes', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ features }),
                });
                
                const result = await response.json();
                
                if (result.success) {
                    const probability = (result.probability * 100).toFixed(2);
                    const alertClass = result.prediction ? 'alert-danger' : 'alert-success';
                    const message = result.prediction 
                        ? `High risk of diabetes detected (${probability}% probability). Please consult a healthcare professional.`
                        : `Low risk of diabetes detected (${probability}% probability). Keep maintaining a healthy lifestyle.`;
                    
                    diabetesResult.querySelector('.alert').className = `alert ${alertClass}`;
                    diabetesResult.querySelector('.alert').textContent = message;
                    diabetesResult.classList.remove('d-none');
                    
                    // Update probability chart
                    updateProbabilityChart('diabetesChart', diabetesChart, result.probability);
                    
                    // Update feature importance chart if available
                    if (result.feature_importance) {
                        updateFeatureImportanceChart('diabetesFeatureChart', diabetesFeatureChart, result.feature_importance);
                    }
                } else {
                    throw new Error(result.error);
                }
            } catch (error) {
                diabetesResult.querySelector('.alert').className = 'alert alert-danger';
                diabetesResult.querySelector('.alert').textContent = 'An error occurred. Please try again.';
                diabetesResult.classList.remove('d-none');
                console.error('Error:', error);
            }
        });
    }
    
    // Cancer detection form handling
    const cancerForm = document.getElementById('cancerForm');
    const cancerResult = document.getElementById('cancerResult');
    let cancerChart = null;
    let cancerFeatureChart = null;
    
    if (cancerForm) {
        cancerForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(cancerForm);
            const features = Array(30).fill(0); // Initialize array for all 30 features
            
            // Fill in the features we have inputs for
            features[0] = parseFloat(formData.get('meanRadius'));
            features[1] = parseFloat(formData.get('meanTexture'));
            features[2] = parseFloat(formData.get('meanPerimeter'));
            features[3] = parseFloat(formData.get('meanArea'));
            
            try {
                const response = await fetch('/predict/cancer', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ features }),
                });
                
                const result = await response.json();
                
                if (result.success) {
                    const probability = (result.probability * 100).toFixed(2);
                    const alertClass = result.prediction ? 'alert-danger' : 'alert-success';
                    const message = result.prediction 
                        ? `Potential malignant indicators detected (${probability}% probability). Immediate medical consultation recommended.`
                        : `No significant malignant indicators detected (${probability}% probability). Regular check-ups recommended.`;
                    
                    cancerResult.querySelector('.alert').className = `alert ${alertClass}`;
                    cancerResult.querySelector('.alert').textContent = message;
                    cancerResult.classList.remove('d-none');
                    
                    // Update probability chart
                    updateProbabilityChart('cancerChart', cancerChart, result.probability);
                    
                    // Update feature importance chart if available
                    if (result.feature_importance) {
                        updateFeatureImportanceChart('cancerFeatureChart', cancerFeatureChart, result.feature_importance);
                    }
                } else {
                    throw new Error(result.error);
                }
            } catch (error) {
                cancerResult.querySelector('.alert').className = 'alert alert-danger';
                cancerResult.querySelector('.alert').textContent = 'An error occurred. Please try again.';
                cancerResult.classList.remove('d-none');
                console.error('Error:', error);
            }
        });
    }
    
    // Heart disease prediction form handling
    const heartForm = document.getElementById('heartForm');
    const heartResult = document.getElementById('heartResult');
    let heartChart = null;
    let heartFeatureChart = null;
    
    if (heartForm) {
        heartForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(heartForm);
            const features = [
                parseFloat(formData.get('age')),
                parseFloat(formData.get('restingBP')),
                parseFloat(formData.get('cholesterol')),
                parseFloat(formData.get('maxHeartRate')),
                parseFloat(formData.get('stDepression')),
                parseInt(formData.get('chestPain')),
                parseInt(formData.get('restECG')),
                parseInt(formData.get('angina')),
                parseInt(formData.get('stSlope')),
                parseInt(formData.get('vessels')),
                parseInt(formData.get('thal'))
            ];
            
            try {
                const response = await fetch('/predict/heart', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ features }),
                });
                
                const result = await response.json();
                
                if (result.success) {
                    const probability = (result.probability * 100).toFixed(2);
                    const alertClass = result.prediction ? 'alert-danger' : 'alert-success';
                    const message = result.prediction 
                        ? `High risk of heart disease detected (${probability}% probability). Please consult a cardiologist.`
                        : `Low risk of heart disease detected (${probability}% probability). Maintain a heart-healthy lifestyle.`;
                    
                    heartResult.querySelector('.alert').className = `alert ${alertClass}`;
                    heartResult.querySelector('.alert').textContent = message;
                    heartResult.classList.remove('d-none');
                    
                    // Update probability chart
                    updateProbabilityChart('heartChart', heartChart, result.probability);
                    
                    // Update feature importance chart if available
                    if (result.feature_importance) {
                        updateFeatureImportanceChart('heartFeatureChart', heartFeatureChart, result.feature_importance);
                    }
                } else {
                    throw new Error(result.error);
                }
            } catch (error) {
                heartResult.querySelector('.alert').className = 'alert alert-danger';
                heartResult.querySelector('.alert').textContent = 'An error occurred. Please try again.';
                heartResult.classList.remove('d-none');
                console.error('Error:', error);
            }
        });
    }
    
    // Reset results when form inputs change
    document.querySelectorAll('form input, form select').forEach(input => {
        input.addEventListener('input', function() {
            const form = this.closest('form');
            const resultDiv = form.id === 'diabetesForm' ? diabetesResult : 
                            form.id === 'cancerForm' ? cancerResult : heartResult;
            resultDiv.classList.add('d-none');
        });
    });
    
    // Model retraining
    const retrainBtn = document.getElementById('retrainBtn');
    if (retrainBtn) {
        retrainBtn.addEventListener('click', async function() {
            try {
                retrainBtn.disabled = true;
                retrainBtn.textContent = 'Retraining...';
                
                const response = await fetch('/retrain', {
                    method: 'POST'
                });
                
                const result = await response.json();
                
                if (result.success) {
                    location.reload(); // Reload page to update stats
                } else {
                    throw new Error(result.error);
                }
            } catch (error) {
                alert('Failed to retrain models: ' + error.message);
            } finally {
                retrainBtn.disabled = false;
                retrainBtn.textContent = 'Retrain Models';
            }
        });
    }
});

// Chart utility functions
function updateProbabilityChart(canvasId, chartInstance, probability) {
    if (chartInstance) {
        chartInstance.destroy();
    }
    
    const ctx = document.getElementById(canvasId).getContext('2d');
    return new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Risk', 'Safe'],
            datasets: [{
                data: [probability * 100, (1 - probability) * 100],
                backgroundColor: [
                    'rgba(255, 99, 132, 0.8)',
                    'rgba(75, 192, 192, 0.8)'
                ]
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Risk Assessment'
                }
            }
        }
    });
}

function updateFeatureImportanceChart(canvasId, chartInstance, featureImportance) {
    if (chartInstance) {
        chartInstance.destroy();
    }
    
    const features = Object.keys(featureImportance);
    const importance = Object.values(featureImportance);
    
    const ctx = document.getElementById(canvasId).getContext('2d');
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: features,
            datasets: [{
                label: 'Feature Importance',
                data: importance,
                backgroundColor: 'rgba(54, 162, 235, 0.8)'
            }]
        },
        options: {
            responsive: true,
            indexAxis: 'y',
            plugins: {
                title: {
                    display: true,
                    text: 'Feature Importance Analysis'
                }
            }
        }
    });
} 