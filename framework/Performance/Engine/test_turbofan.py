import numpy as np
import matplotlib.pyplot as plt
from framework.Performance.Engine.engine_performance import turbofan

# Análise variando Bypass Ratio e Fan Diameter

# Análise ajustada para comparar em um único gráfico ao variar os parâmetros

# Análise explorando mais grandezas: consumo de combustível e eficiência específica

def analyze_additional_parameters():
    # Parâmetros de teste
    altitudes = [0, 20000, 40000]  # Altitudes fixas
    mach = 0.7  # Número de Mach fixo
    throttle = 1.0  # Posição do acelerador máxima
    bypass_ratios = [4, 6, 8, 10]  # Diferentes razões de bypass
    fan_diameters = [1.6, 1.8, 2.0, 2.2]  # Diferentes diâmetros do fan

    # Comparação para diferentes bypass ratios (foco em consumo de combustível e eficiência específica)
    plt.figure(figsize=(10, 6))
    for bypass in bypass_ratios:
        fuel_flows = []
        efficiencies = []
        for altitude in altitudes:
            vehicle = {
                "engine": {
                    "fan_pressure_ratio": 1.7,
                    "compressor_pressure_ratio": 30,
                    "bypass": bypass,
                    "fan_diameter": 1.8,  # Diâmetro fixo
                    "turbine_inlet_temperature": 1600,
                    "fan_rotation_ref": 5000,
                    "compressor_rotation_ref": 15000,
                }
            }
            force, fuel_flow, _ = turbofan(altitude, mach, throttle, vehicle)
            fuel_flows.append(fuel_flow)
            efficiencies.append(force / fuel_flow)  # Eficiência específica (empuxo por consumo de combustível)
        
        # Plotar consumo de combustível
        plt.plot(altitudes, fuel_flows, label=f"Bypass Ratio: {bypass}")
    plt.title("Consumo de Combustível vs Altitude para diferentes Bypass Ratios")
    plt.xlabel("Altitude [ft]")
    plt.ylabel("Consumo de Combustível [kg/hr]")
    plt.grid()
    plt.legend()
    plt.show()

    # Comparação de eficiência específica
    plt.figure(figsize=(10, 6))
    for bypass in bypass_ratios:
        efficiencies = []
        for altitude in altitudes:
            vehicle = {
                "engine": {
                    "fan_pressure_ratio": 1.7,
                    "compressor_pressure_ratio": 30,
                    "bypass": bypass,
                    "fan_diameter": 1.8,
                    "turbine_inlet_temperature": 1600,
                    "fan_rotation_ref": 5000,
                    "compressor_rotation_ref": 15000,
                }
            }
            force, fuel_flow, _ = turbofan(altitude, mach, throttle, vehicle)
            efficiencies.append(force / fuel_flow)
        plt.plot(altitudes, efficiencies, label=f"Bypass Ratio: {bypass}")
    plt.title("Eficiência Específica vs Altitude para diferentes Bypass Ratios")
    plt.xlabel("Altitude [ft]")
    plt.ylabel("Eficiência Específica (Empuxo/Consumo de Combustível) [N/(kg/hr)]")
    plt.grid()
    plt.legend()
    plt.show()

    # Comparação para diferentes fan diameters
    plt.figure(figsize=(10, 6))
    for fan_diameter in fan_diameters:
        fuel_flows = []
        efficiencies = []
        for altitude in altitudes:
            vehicle = {
                "engine": {
                    "fan_pressure_ratio": 1.7,
                    "compressor_pressure_ratio": 30,
                    "bypass": 6,
                    "fan_diameter": fan_diameter,
                    "turbine_inlet_temperature": 1600,
                    "fan_rotation_ref": 5000,
                    "compressor_rotation_ref": 15000,
                }
            }
            force, fuel_flow, _ = turbofan(altitude, mach, throttle, vehicle)
            fuel_flows.append(fuel_flow)
            efficiencies.append(force / fuel_flow)
        
        # Plotar consumo de combustível
        plt.plot(altitudes, fuel_flows, label=f"Fan Diameter: {fan_diameter:.1f} m")
    plt.title("Consumo de Combustível vs Altitude para diferentes Fan Diameters")
    plt.xlabel("Altitude [ft]")
    plt.ylabel("Consumo de Combustível [kg/hr]")
    plt.grid()
    plt.legend()
    plt.show()

    # Comparação de eficiência específica para fan diameters
    plt.figure(figsize=(10, 6))
    for fan_diameter in fan_diameters:
        efficiencies = []
        for altitude in altitudes:
            vehicle = {
                "engine": {
                    "fan_pressure_ratio": 1.7,
                    "compressor_pressure_ratio": 30,
                    "bypass": 6,
                    "fan_diameter": fan_diameter,
                    "turbine_inlet_temperature": 1600,
                    "fan_rotation_ref": 5000,
                    "compressor_rotation_ref": 15000,
                }
            }
            force, fuel_flow, _ = turbofan(altitude, mach, throttle, vehicle)
            efficiencies.append(force / fuel_flow)
        plt.plot(altitudes, efficiencies, label=f"Fan Diameter: {fan_diameter:.1f} m")
    plt.title("Eficiência Específica vs Altitude para diferentes Fan Diameters")
    plt.xlabel("Altitude [ft]")
    plt.ylabel("Eficiência Específica (Empuxo/Consumo de Combustível) [N/(kg/hr)]")
    plt.grid()
    plt.legend()
    plt.show()

analyze_additional_parameters()
