import pickle

F1_GCD = [-2141506.0 ,
-2091406.0 ,
-1993062.0,
-1970112.0,
-1966039.0,
-1962015.0,
-1578819.0,
-1571703.0]

F2_GCD = [ 74273.384526,
74018.884933,
73589.696445,
72231.429727,
72204.241456,
70573.472955,
69606.309884,
69351.810291]
with open("Database/Results_Multi_Optim/functions/case8_cost_pareto.pkl", "wb") as f:   #Pickling
    pickle.dump(F2_GCD, f)