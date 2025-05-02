# F1 Shanghai Grand Prix 2025 Prediction Model

A machine learning-based prediction system for forecasting Formula 1 race results for the 2025 Shanghai Grand Prix using historical F1 data and team/driver characteristics.

![Starting Grid vs Predicted Finish](grid_vs_finish.png)

## Project Overview

This project uses historical Formula 1 data from the 2022-2024 seasons to build a predictive model that forecasts the finishing positions for the upcoming 2025 Shanghai Grand Prix. The model incorporates:

- Historical driver performance
- Team strength assessment
- Qualifying position influence
- Driver experience factors
- Circuit-specific performance patterns

The system handles team changes for 2025 (like Hamilton moving to Ferrari) and accommodates rookies through team performance metrics.

## Key Features

- **Data Collection**: Automated fetching of historical F1 race data using the FastF1 API
- **Feature Engineering**: Comprehensive driver and team metrics creation
- **Machine Learning**: Random Forest regression model to predict race finishing positions
- **Team Change Handling**: Sophisticated method for handling 2025 driver lineup changes
- **Visualization**: Three different visualizations of prediction results
- **Fallback Systems**: Robust data generation when API data is incomplete

## Visualizations

### 1. Grid Position vs Predicted Finish

![Starting Grid vs Predicted Finish](grid_vs_finish.png)

This visualization shows:

- How each driver is expected to perform relative to their qualifying position
- Points below the diagonal line indicate drivers expected to finish better than their starting position
- Points above the line show drivers predicted to lose positions during the race

### 2. Driver Performance Prediction

![Expected Finishing Position](shanghai_gp_prediction.png)

This chart displays:

- Expected finishing position for each driver
- Color-coded by team
- Lower values indicate better predicted performance

### 3. Team Performance Prediction

![Team Performance](team_performance.png)

This visualization shows:

- Average predicted finishing position by team
- Lower values indicate stronger team performance
- Teams are ranked from strongest to weakest

## Methodology

1. **Data Collection**

   - Historical race results from 2022-2024 seasons
   - Driver and team mappings
   - Circuit-specific performance patterns

2. **Feature Engineering**

   - Grid position influence
   - Team performance metrics
   - Driver experience quantification
   - Circuit-specific indicators

3. **Machine Learning Model**

   - Random Forest Regressor trained on historical finishing positions
   - Feature importance analysis to understand prediction factors
   - Position randomization with team-weighted probabilities for qualifying simulation

4. **Prediction Generation**
   - Qualifying simulation based on team strength
   - Race position prediction using the trained model
   - Analysis of expected position changes during the race

## Technologies Used

- **Python**: Core programming language
- **FastF1**: Formula 1 data access API
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib/Seaborn**: Data visualization
- **NumPy**: Numerical computing

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages (install via pip):
  ```
  pip install fastf1 pandas numpy scikit-learn matplotlib seaborn
  ```

### Installation

1. Clone this repository:

   ```
   git clone https://github.com/yourusername/f1-shanghai-prediction.git
   cd f1-shanghai-prediction
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Run the prediction script:
   ```
   python shanghai_f1.py
   ```

## Results

The model predicts a podium of:

1. 🥇 Max Verstappen (Red Bull Racing)
2. 🥈 Liam Lawson (Red Bull Racing)
3. 🥉 George Russell (Mercedes)

The full prediction includes expected finishing positions for all 20 drivers competing in the 2025 Shanghai Grand Prix.

## Future Improvements

- Weather condition impact modeling
- Tire strategy optimization simulation
- Driver head-to-head performance analytics
- Race incident probability modeling
- Real-time data integration during race weekends

## Resources

- [FastF1 Documentation](https://theoehrly.github.io/Fast-F1/)
- [Formula 1 Official Website](https://www.formula1.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

