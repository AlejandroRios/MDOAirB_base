
import pickle

with open('Database/Family/40_to_100/all_dictionaries/'+str(54)+'.pkl', 'rb') as f:
    all_info_acft1 = pickle.load(f)

# # with open('Database/Family/101_to_160/all_dictionaries/'+str(13)+'.pkl', 'rb') as f:
# #     all_info_acft1 = pickle.load(f)

# with open('Database/Family/161_to_220/all_dictionaries/'+str(0)+'.pkl', 'rb') as f:
#     all_info_acft1 = pickle.load(f)

# with open('Database/Family_DD/40_to_100/all_dictionaries/'+str(18)+'.pkl', 'rb') as f:
#     all_info_acft1 = pickle.load(f)

# with open('Database/Family_DD/101_to_160/all_dictionaries/'+str(28)+'.pkl', 'rb') as f:
#     all_info_acft1 = pickle.load(f)

# with open('Database/Family_DD/161_to_220/all_dictionaries/'+str(53)+'.pkl', 'rb') as f:
#     all_info_acft1 = pickle.load(f)


print('Design parameters:')
print('pax:',all_info_acft1['vehicle']['aircraft']['passenger_capacity'])
print('seat abst:',all_info_acft1['vehicle']['fuselage']['seat_abreast_number'])


print('area:',all_info_acft1['vehicle']['wing']['area'])
print('AR:',all_info_acft1['vehicle']['wing']['aspect_ratio'])
print('break:',all_info_acft1['vehicle']['wing']['semi_span_kink'])
print('sweep:',all_info_acft1['vehicle']['wing']['sweep_c_4'])

print('TR:',all_info_acft1['vehicle']['wing']['taper_ratio'])
print('Twist:',all_info_acft1['vehicle']['wing']['twist'])

print('type:',all_info_acft1['vehicle']['engine']['type'])
print('BPR:',all_info_acft1['vehicle']['engine']['bypass'])
print('Fan D:',all_info_acft1['vehicle']['engine']['fan_diameter'])
print('Tmax:',float(all_info_acft1['vehicle']['engine']['maximum_thrust']))
print('TiT:',all_info_acft1['vehicle']['engine']['turbine_inlet_temperature'])
print('Range:',all_info_acft1['vehicle']['performance']['range'])

print('MTOW:',all_info_acft1['vehicle']['aircraft']['maximum_takeoff_weight'])
print('MLW:',all_info_acft1['vehicle']['aircraft']['maximum_landing_weight'])
print('MZFW:',all_info_acft1['vehicle']['aircraft']['maximum_zero_fuel_weight'])
print('OEW:',all_info_acft1['vehicle']['aircraft']['operational_empty_weight'])
# print('WF:',all_info_acft1['vehicle']['aircraft']['maximum_fuel_capacity'])