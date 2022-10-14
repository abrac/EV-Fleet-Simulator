




# This function intakes a pandas dataframe of trip data ('journey'), 
# and adds columns with the geodesic distance and slope angles between consecutive observations

def getDistSlope(journey):
    Distance = np.zeros(len(journey))
    Slope = np.zeros(len(journey))
    l_route = journey.shape[0]

    for i in range(0,l_route-1): # end before i + 1 out of range
        elev_change = journey.Altitude.iloc[i+1] - journey.Altitude.iloc[i]
      #  elev_change = journey['ElevChange'][i] 
        dist_lateral = geopy.distance.geodesic(journey.Coordinates.iloc[i],  # Lateral distance in meters - dist between two lat/lon coord pairs
                                       journey.Coordinates.iloc[i+1]).m  #Coordinates is list(zip(journey['Latitude'], journey['Longitude']))

        dist_3d = np.sqrt(dist_lateral**2 + elev_change**2)  # geodesic distance (3d = accounting for elevation), in meters
        if i == 0:
            Distance[i] = 0
        else:
            Distance[i] = dist_3d
        if Distance[i] != 0 and elev_change != 0:
            Slope[i] = np.arcsin(elev_change/dist_3d) #calculate slope angle in radians from opposite/hypotenuse

    journey['Displacement_m'] = list(Distance) #add displacement to dataframe
    journey['slope_rad'] = list(Slope) #add slope in radians to dataframe
    journey['slope_deg'] = journey['slope_rad'] * 180/np.pi #calculate slope angle in degrees



######################## The three drag forces in the model - used in Vehicle class below #################################
#Inputs described in Vehicle class 


# Road load (friction) (N)
def getRoadFriction(mass, c_rr, slope, vel, grav = 9.81): #c_rr is coeff of rolling resistance
    rf = 0
    if vel > 0.3:
        rf = -mass * grav * c_rr * np.cos(slope)
    return rf
 
# Aerodynamic Drag Force (N)
def getAerodynamicDrag(c_d, A, vel, rho = 1.184): # rho is air density 20C, c_d is air drag coeff
    return -0.50 * rho * c_d * A * vel**2

# Road Slope Force (N)
def getRoadSlopeDrag(mass, slope, grav = 9.81):
    return -mass * grav * np.sin(slope)
################################################################################################ 



class Drivecycle: 
    """
    Inputs: dataframe with journey info
    Outputs: drivecycle class
    """
    def __init__(self, journey): 
        self.displacement = journey.Displacement_m # m
        self.velocity = journey.Velocity # m/s
        self.slope = journey.slope_rad # rad
        self.time = journey.RelTime.max() # Total Time Elapsed
        self.dt = journey.DeltaT # Time elapsed between each timestamp
        self.dv = journey.DeltaV # Difference in velocity between two consectuive timesteps
        self.acceleration = journey.Acceleration # dv/dt

################################################################################################ 



class Vehicle:
    """
    Inputs: Physical parameters of vehicle for modeling energy consumption
    Returns an array of vehicle energy consumption at each timestamp
    
    mass - vehicle mass (kg)
    cd - coefficient of drag
    crr - coefficient of rolling resistance
    A - vehicle frontal area (m^2)
    eff - vehicle propulsion efficiency
    rgbeff - regenerative braking energy recuperation efficiency 
    cap - vehicle battery cap (kWh)
    p0 - constant power intake (W)
    """
    
    def __init__(self, mass = 3900, payload = 0,
                 cd = 0.36, crr = 0.02, A = 4,
                 eff = 0.9, rgbeff = 0.65, cap = 100, p0 = 100):

        # Vehicle physical parameters
        self.mass = mass # kg
        self.load = payload #kg
        self.crr = crr # coefficient of rolling resistance
        self.cd = cd  # air drag coefficient
        self.A = A # m^2, Approximation of vehicle frontal area 
        self.eff = eff # %, powertrain efficiency
        self.rgbeff = rgbeff  #%, regen brake efficiency
       # self.capacity = cap  # not used
       # self.battery = cap # not used
        self.p0 = p0 # constant power loss in W (to run the vehicle besides driving)
    
    def getEnergyExpenditure(self,cycle,regbrake = True):
        # computes energy expenditure from a Drivecycle object
        # dt default 1 second
        
        v = cycle.velocity # m/s 
        s = cycle.slope # rad
        a = cycle.acceleration # m/s^2
        dt = cycle.dt # s
        d = cycle.displacement # m
        dv = cycle.dv # m/s 
        
        if regbrake == True:
            RGBeff = self.rgbeff # static regen coeff
        else:
            RGBeff = 0
            
                        


        Er = [] # Total energy consumption (J), from battery if positive, to battery if negative (RGbrake)

        Frpb = [] # force propulsive or braking

        # Drag forces
        Fa = [] # aerodynamic drag
        Frr = [] # rolling resistance
        Fhc = [] # hill climb (slope drag)
        Fr = [] # Drag force: Inertia + Friction + Aero Drag + Slope Drag (N)

        

        
        for slope,vel,acc,delta_t, delta_v in zip(s,v,a,dt, dv):
            
            if vel == 0:
                force, frr, fa, fhc = 0,0,0,0
            else:
                frr = getRoadFriction(self.mass,self.crr, slope, vel)
                fa = getAerodynamicDrag(self.cd, self.A, vel)
                fhc = getRoadSlopeDrag(self.mass, slope)
                
            Frr.append(frr)
            Fa.append(fa)
            Fhc.append(fhc)

            force = frr + fa + fhc # (N + N + N)  - total drag force
            
            exp_speed_delta = force * delta_t / self.mass # (N) * (s) / (kg) 
            
            unexp_speed_delta = delta_v - exp_speed_delta # (m/s) - (m/s) 
            
            try:
                prop_brake_force = unexp_speed_delta / delta_t * self.mass #(m/s) / (s) * (kg) = N
            
                kinetic_power = prop_brake_force * vel #(N) * (m/s) = (W)
                
                propultion_work = kinetic_power * delta_t  # (W) * (s) -- kinetic energy
                
            except ZeroDivisionError: 
                prop_brake_force = 0
                kinetic_power = 0
                propultion_work = 0


            Fr.append(force) # N
            Frpb.append(prop_brake_force) # N
            Er.append(propultion_work) # Ws

        
        ErP = [0.0]*len(Er)
        ErB = [0.0]*len(Er)
        offtake_power = [0.0]*len(Er)

        for i in range(len(Er)):
            offtake_power[i] = self.p0 # constant offtake

            if Er[i] > 0: # energy that is used for propulsion 
                ErP[i] = Er[i]/self.eff 
            
            elif Er[i] < 0: #energy that is regen'd back into the battery
                ErB[i]= Er[i]*RGBeff   

    
    
                # output in watt-hours
        return Fa, Frr, Fhc, Fr,\
    Frpb,
    Er,ErP, ErB,\
    offtake_power
    #N,N,N,N,N,m/s,ms,N,Ws, Ws,Ws,Ws


        