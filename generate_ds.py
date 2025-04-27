import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Number of rows
num_samples = 100000

# Generate realistic features
industrial_output = np.random.normal(500, 100, num_samples)  # Industrial production index
energy_consumption = np.random.normal(300, 50, num_samples)  # Energy usage in GWh
transport_emissions = np.random.normal(200, 30, num_samples)  # Transportation CO2 emissions (in metric tons)
population_density = np.random.normal(1000, 200, num_samples)  # People per square km
weather_temp = np.random.normal(25, 5, num_samples)  # Temperature in Celsius
weather_humidity = np.random.normal(60, 10, num_samples)  # Humidity percentage
renewable_energy_share = np.random.uniform(10, 50, num_samples)  # % of energy from renewable sources
carbon_tax = np.random.uniform(5, 30, num_samples)  # Carbon tax per ton of CO2
energy_efficiency = np.random.uniform(50, 100, num_samples)  # Efficiency rating (0-100)
traffic_index = np.random.normal(70, 15, num_samples)  # Congestion index
forest_cover = np.random.uniform(20, 80, num_samples)  # % of land covered by forest
industrial_waste = np.random.normal(100, 20, num_samples)  # Industrial waste in tons
urbanization_rate = np.random.uniform(30, 80, num_samples)  # % of population in urban areas

# Generate realistic target variable (CO2 emissions) with a correlation to features
co2_emissions = (
    industrial_output * 0.5 +
    energy_consumption * 0.3 +
    transport_emissions * 0.4 -
    renewable_energy_share * 2 -
    carbon_tax * 1.5 -
    energy_efficiency * 0.7 -
    forest_cover * 1.2 +
    industrial_waste * 0.8 +
    np.random.normal(0, 10, num_samples)  # Adding some noise
)

# Create DataFrame
df = pd.DataFrame({
    'industrial_output': industrial_output,
    'energy_consumption': energy_consumption,
    'transport_emissions': transport_emissions,
    'population_density': population_density,
    'weather_temp': weather_temp,
    'weather_humidity': weather_humidity,
    'renewable_energy_share': renewable_energy_share,
    'carbon_tax': carbon_tax,
    'energy_efficiency': energy_efficiency,
    'traffic_index': traffic_index,
    'forest_cover': forest_cover,
    'industrial_waste': industrial_waste,
    'urbanization_rate': urbanization_rate,
    'co2_emissions': co2_emissions
})

# Save dataset to CSV
df.to_csv("emissions_dataset.csv", index=False)

print("Dataset generated and saved as emissions_dataset.csv")
