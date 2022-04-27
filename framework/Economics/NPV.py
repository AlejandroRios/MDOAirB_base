
def aircraft_price(MTOW,share):
    # RefPrice=23000000
    # MTOWfac=MTOW/22000
    # WTADJPrice=RefPrice*MTOWfac
    # Price     =(1+(0.3-0.5*Share))*WTADJPrice

    Yref=0.3
    Xref=0.6

    # Model derived from Airbus price list (Narrow Bodies,1 aisle) @ 23/12/2018
    y0=3.3383
    mu=0.0013
    WTADJPrice=y0+mu*MTOW 
    DESC      =Yref/Xref*share
    Price     =WTADJPrice*(1-DESC)
    y=Price*1E6
    return y


def NPV(Share,IR,p,MTOW,wS,ne,Tmax,ediam,NPax,KVA):
    ###########################################################################
    #MAIN DATA

    ##Baseline Aircraft
    Baseline = {}
    Baseline['MTOW']  = 22010
    Baseline['wS'] = 52
    Baseline['ne'] = 2
    Baseline['ediam'] = 1.52
    Baseline['NPax'] = 50
    Baseline['Tmax'] = 8895
    Baseline['KVA'] = 75
    MTOWfac  = MTOW/Baseline['MTOW']
    Tfac   = Tmax/Baseline['Tmax']
    wS_ft2 = wS*(3.28)**2
    ediam_in = ediam*39.37
    
    ##Market info
    Ref_Share =0.6
    Ref_Price =23000000
    Sharefac  =Share/Ref_Share
    Price     =aircraft_price(MTOW,Share)

    ##Deliveries Forecast for Reference share

    deliveries_list = [0, 0, 0, 0, 6, 20, 44, 92, 94, 102, 94, 90, 84, 30, 10]

    deliveries = []
    for i in range(len(deliveries_list)):
        if i > 3:
            deliveries_aux = share_factor*deliveries_list[i]
        else:
            deliveries_aux = deliveries_list[i]

        deliveries.append(deliveries_aux)



    ##Program share
    manufacturer = 0.7
    partner = 1-manufacturer
    man_power_cost_manufacturer = 41.72
    man_power_cost_partner = 83.44
    man_power_cost_average = manufacturer * \
        man_power_cost_manufacturer + partner*man_power_cost_partner

    ##Product Development Cost - Man Power Distribution
    # Man Power Distribution
    man_power_distribution = [0.25, 0.35, 0.30, 0.1]
    # Infrastructure Distribution
    infrastructure_distribution = [0.55, 0.25, 0.15, 0.05]
    # Generic Cost Distribution
    generic_cost_distribution = [0.25, 0.35, 0.40, 0.1]


    ##Production Cost - Complexity factor
    complexity_factors = {}
    complexity_factors['landing_gear'] = 1.2
    complexity_factors['flight_controls'] = 1.4
    complexity_factors['engines'] = 1.1
    complexity_factors['nacelles'] = 1.0
    complexity_factors['interior'] = 1.0
    complexity_factors['electrical'] = 1.3
    complexity_factors['avionics'] = 0.9
    complexity_factors['structures'] = 1.5
    complexity_factors['fuel'] = 1.0
    ###########################################################################
    #NON-RECURRING COSTS (Development)
    ###########################################################################

    ##Man Power - Management and Administration hours
    MAA_H.Prelim_design_baseline=27983
    MAA_H.Interior_baseline     =229950
    MAA_H.Structures_baseline   =279833
    MAA_H.Electrics_baseline    =365000
    MAA_H.Mechanics_baseline    =121667
    MAA_H.Hidraulics_baseline    =60833
    MAA_H.Propulsion_baseline    =51100

    MAA_H    = MAA_H.Prelim_design_baseline+...
            MAA_H.Interior_baseline*MTOWfac+...
            MAA_H.Structures_baseline*MTOWfac+...
            MAA_H.Electrics_baseline*MTOWfac+...
            MAA_H.Mechanics_baseline*MTOWfac+...
            MAA_H.Hidraulics_baseline*MTOWfac+...
            MAA_H.Propulsion_baseline*Tfac

    ##Man Power - Engineering hours
    ENG_H.Aero_baseline      =135050
    ENG_H.Aeroe_baseline     =237250
    ENG_H.Perfo_baseline     =76650
    ENG_H.Stru_baseline      =365000
    ENG_H.Interior_baseline  =92467
    ENG_H.Elec_baseline      =365000
    ENG_H.Mechanics_baseline =399067
    ENG_H.Hydraulics_baseline=97333
    ENG_H.Propulsion_baseline=121667
    ENG_H.Certif_baseline    =121667
    ENG_H.Config_baseline    =62050
    ENG_H.Docs_baseline      =115583

    ENG_H    = ENG_H.Aero_baseline*MTOWfac +...
            ENG_H.Aeroe_baseline*MTOWfac +...
            ENG_H.Perfo_baseline*MTOWfac +...
            ENG_H.Stru_baseline*MTOWfac +...
            ENG_H.Interior_baseline*MTOWfac +...
            ENG_H.Elec_baseline*MTOWfac +...
            ENG_H.Mechanics_baseline*MTOWfac +...
            ENG_H.Hydraulics_baseline*MTOWfac +...
            ENG_H.Propulsion_baseline*Tfac +...
            ENG_H.Certif_baseline*MTOWfac+...
            ENG_H.Config_baseline+...
            ENG_H.Docs_baseline
            
    ##Man Power - Test and Validation
    TEST_H.Ground_testing_baseline        =208050
    TEST_H.Systems_testing_baseline       =111933
    TEST_H.Flight_test_baseline           =160600
    TEST_H.Instrumentation_baseline       =122883
    TEST_H.Pilots_baseline                =48667
    TEST_H.Prototypes_baseline            =2828750
    TEST_H.Rigs_baseline                  =60833
    TEST_H.Body_baseline                  =97333
    TEST_H.Prot_Maintenance_baseline      =585217
    TEST_H.Manufacturing_Tooling_baseline =526817
    TEST_H.MethodsProcess_baseline        =784750
    TEST_H.Quality_Control_baseline       =138700

    TEST_H    = TEST_H.Ground_testing_baseline*MTOWfac +...
                TEST_H.Systems_testing_baseline*MTOWfac +...
                TEST_H.Flight_test_baseline*MTOWfac +...
                TEST_H.Instrumentation_baseline*MTOWfac +...
                TEST_H.Pilots_baseline*MTOWfac +...
                TEST_H.Prototypes_baseline*MTOWfac +...
                TEST_H.Rigs_baseline*MTOWfac +...
                TEST_H.Body_baseline*MTOWfac +...
                TEST_H.Prot_Maintenance_baseline*MTOWfac +...
                TEST_H.Manufacturing_Tooling_baseline*MTOWfac +...
                TEST_H.MethodsProcess_baseline+...
                TEST_H.Quality_Control_baseline
            
    ##Man Power - Support hours
    SUP_H.ServBul_baseline       =19467
    SUP_H.SpareParts_baseline    =103417
    SUP_H.OpsManuals_baseline    =203183
    SUP_H.CustServ_baseline      =6083
    SUP_H.Documentation_baseline =128967
    SUP_H.TechSupp_baseline      =77867
    SUP_H.MaintPlan_baseline     =96117

    SUP_H     = SUP_H.ServBul_baseline+...
                SUP_H.SpareParts_baseline*MTOWfac+...
                SUP_H.OpsManuals_baseline+...
                SUP_H.CustServ_baseline+...
                SUP_H.Documentation_baseline+...
                SUP_H.TechSupp_baseline+...
                SUP_H.MaintPlan_baseline       
            
    TOT_H  = MAA_H+ENG_H+TEST_H+SUP_H  
    COST_H = TOT_H*Man_Power_Cost.AVG

    for i=1:4
    MP_COST_Y(i)=COST_H*PDC_MPD(i)
    end  
    MP_COST_Y=MP_COST_Y'

    ##Infrastructure Cost
    INFR.MachineEquip_baseline      =17880000
    INFR.FSW_baseline               =9536000
    INFR.Prototypes_baseline        =35760000
    INFR.AssemblyLine_baseline      =1192000
    INFR.ComputerResources_baseline =6913600
    INFR.Instruments_baseline       =596000
    INFR.OfficeSupplies_baseline    =596000
    INFR.Software_baseline          =1788000
    INFR.Others_baseline            =596000

    INFR_COST  = INFR.MachineEquip_baseline*MTOWfac +...
                INFR.FSW_baseline*MTOWfac +...
                INFR.Prototypes_baseline*MTOWfac +...
                INFR.AssemblyLine_baseline*MTOWfac +...
                INFR.ComputerResources_baseline+...
                INFR.Instruments_baseline+...
                INFR.OfficeSupplies_baseline+...
                INFR.Software_baseline+...
                INFR.Others_baseline
            
    for i=1:4
    INFRA_COST_Y(i)=INFR_COST*PDC_INFRAD(i)
    end          
    INFRA_COST_Y=INFRA_COST_Y'

    ##General Costs
    GEN.Communications_baseline     =834400
    GEN.InstrFT_baseline            =4529600
    GEN.Tooling_baseline            =2980000
    GEN.Traveling_baseline          =5006400
    GEN.TPMP_baseline               =23840000
    GEN.TPMPP_baseline              =27416000
    GEN.Fuel_baseline               =4872900

    GEN_COST  =  GEN.Communications_baseline+...
                GEN.InstrFT_baseline+...
                GEN.Tooling_baseline+...
                GEN.Traveling_baseline+...
                GEN.TPMP_baseline+...
                GEN.TPMPP_baseline+...
                GEN.Fuel_baseline*MTOWfac

    for i=1:4
    GEN_COST_Y(i)=GEN_COST*PDC_GEND(i)
    end          
    GEN_COST_Y=GEN_COST_Y'

    ##TOTAL NON RECURRING COST PER YEAR
    for i=1:4
    NRC(i)=(MP_COST_Y(i)+INFRA_COST_Y(i)+GEN_COST_Y(i))*MShare.Manufacturer
    end
    NRC=NRC'


    ###########################################################################
    #RECURRING COSTS (Production)
    ###########################################################################

    ##MATERIAL COSTS

    MAT.Landing_Gear      =8
    MAT.Flight_Controls   =834
    MAT.Engines          =2000000
    MAT.Naceles          =11920
    MAT.Interior         =19072
    MAT.Electrical_System =8940
    MAT.Avionics         =953600
    MAT.Structures       =143
    MAT.Fuel             =119

    MAT_COST = MAT.Landing_Gear*CF.Landing_Gear*MTOW+...
            MAT.Flight_Controls*CF.Flight_Controls*wS_ft2+...
            MAT.Engines*CF.Engines*ne*Tfac+...
            MAT.Naceles*CF.Naceles*ediam_in+...
            MAT.Interior*CF.Interior*NPax+...
            MAT.Electrical_System*CF.Electrical*KVA+...
            MAT.Avionics*CF.Avionics+...
            MAT.Structures*CF.Structures*MTOW+...
            MAT.Fuel*CF.Fuel*wS_ft2
                
    ##MANPOWER COST
    LCPH           =25
    MPH.Baseline(5)=36714
    MPH.Baseline(6)=26452
    MPH.Baseline(7)=23363
    MPH.Baseline(8)=21451
    MPH.Baseline(9)=20218
    MPH.Baseline(10)=19416
    MPH.Baseline(11)=18899
    MPH.Baseline(12)=18581
    MPH.Baseline(13)=18389
    MPH.Baseline(14)=18281
    MPH.Baseline(15)=18173

    for i=5:p
    RMP_COST(i)=MPH.Baseline(i)*Del(i)*MTOWfac*LCPH
    end
    RMP_COST=RMP_COST'

    #COST MATRIX

    for i=1:p
        if i<=4
        COST(i,1)=(MP_COST_Y(i,1)+INFRA_COST_Y(i,1)+GEN_COST_Y(i,1))*MShare.Manufacturer        
        else
        C1=Del(i)*MAT_COST
        C2=RMP_COST(i,1)
        C3=(C1+C2)*0.17
        C4=(C1+C2)*0.08
        C5=C1*0.05
        COST(i,1)=C1+C2+C3+C4+C5
        end    
    end    

    #CASH FLOW MATRIX
    TOT_NPV=0
    for i=1:p
    SALES=Del(i)*Price 
    CASHFLOW(i)=SALES-COST(i)   
    PV(i)=CASHFLOW(i)/((1+IR)**(i-1))
    TOT_NPV=TOT_NPV+PV(i)
    end

    PV=PV'
    CASHFLOW=CASHFLOW'
    IRR=IRRCALC(CASHFLOW)

    #Break even calculation
    BE=0
    for i=2:p
        X1=i-1
        Y1=PV(i-1)
        X2=i
        Y2=PV(i)
        X =X1-Y1*(X2-X1)/(Y2-Y1)
        if Y1*Y2<0
            BE=X
            break
        end
    end




    return TOT_NPV,CASHFLOW,PV,IRR,BE,Price



    


