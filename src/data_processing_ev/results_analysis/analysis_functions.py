from data_processing_ev.ev_simulation.hull_ev_simulation import \
    DFLT_INTEGRATION_MTHD, INTEGRATION_MTHD  # TODO Deprecate!
import numpy as np
import geopy.distance


def getDistSlope(journey, integration_mthd=DFLT_INTEGRATION_MTHD, geo=True, **kwargs):
    """
    Inputs:
        - dataframe with vehicle journey data
        - geo: If true, then the coordinates are interpreted as latitude and
            longitude. If false, then the coordinates are interpreted as coordinates in
            meters (with respect to some reference).

    Calculates distance between each successive pair of coordinates,
    accounting also for elevation difference.

    Calculates slope angle faced by vehicle at each timestamp.
    """

    if 'vehicle_z' not in journey.columns:
        journey['vehicle_z'] = 0

    Distance = np.zeros(len(journey))
    Slope = np.zeros(len(journey))
    l_route = journey.shape[0]

    if integration_mthd == INTEGRATION_MTHD['fwd']:
        Distance[-1] = 0
        Slope[-1] = 0
        for i in range(0,l_route-1):  # end before i + 1 out of range
            elev_change = journey['vehicle_z'].iloc[i+1] - journey['vehicle_z'].iloc[i]
            if geo:
                dist_lateral = geopy.distance.geodesic(
                    journey.Coordinates.iloc[i],
                        # Lateral distance in meters - dist between two lat/lon
                        # coord pairs
                    journey.Coordinates.iloc[i+1]).m
                        # Coordinates is list(zip(journey['Latitude'],
                        # journey['Longitude']))
            else:
                dist_lateral = np.sqrt(
                    (journey.Coordinates.iloc[i+1][0] - journey.Coordinates.iloc[i][0])**2 +
                    (journey.Coordinates.iloc[i+1][1] - journey.Coordinates.iloc[i][1])**2)

            dist_3d = np.sqrt(dist_lateral**2 + elev_change**2)
                # geodesic distance (3d = accounting for elevation), in meters
            Distance[i] = dist_3d
            if Distance[i] != 0 and elev_change != 0:
                Slope[i] = np.arcsin(elev_change/dist_3d)
                    # calculate slope angle in radians from opposite/hypotenuse

    elif integration_mthd == INTEGRATION_MTHD['bwd']:
        Distance[0] = 0
        Slope[0] = 0
        for i in range(1,l_route):  # Start after i-1 out of range
            elev_change = journey['vehicle_z'].iloc[i] - journey['vehicle_z'].iloc[i-1]
            if geo:
                dist_lateral = geopy.distance.geodesic(
                    journey.Coordinates.iloc[i-1],
                        # Lateral distance in meters - dist between two lat/lon
                        # coord pairs
                    journey.Coordinates.iloc[i]).m
                        # Coordinates is list(zip(journey['Latitude'],
                        # journey['Longitude']))
            else:
                dist_lateral = np.sqrt(
                    (journey.Coordinates.iloc[i][0] - journey.Coordinates.iloc[i-1][0])**2 +
                    (journey.Coordinates.iloc[i][1] - journey.Coordinates.iloc[i-1][1])**2)

            dist_3d = np.sqrt(dist_lateral**2 + elev_change**2)
                # geodesic distance (3d = accounting for elevation), in meters
            Distance[i] = dist_3d
            if Distance[i] != 0 and elev_change != 0:
                Slope[i] = np.arcsin(elev_change/dist_3d)
                    # calculate slope angle in radians from opposite/hypotenuse

    elif integration_mthd == INTEGRATION_MTHD['ctr']:
        Distance[0] = 0
        Distance[-1] = 0
        Slope[0] = 0
        Slope[-1] = 0
        for i in range(1, l_route-1):  # Start after i-1 out of range
            elev_change = (journey['vehicle_z'].iloc[i+1] - journey['vehicle_z'].iloc[i-1])/2
            if geo:
                dist_lateral = (geopy.distance.geodesic(
                    journey.Coordinates.iloc[i-1],
                        # Lateral distance in meters - dist between two lat/lon
                        # coord pairs
                    journey.Coordinates.iloc[i+1]).m) / 2
                        # Coordinates is list(zip(journey['Latitude'],
                        # journey['Longitude']))
            else:
                dist_lateral = (np.sqrt(
                    (journey.Coordinates.iloc[i+1][0] -
                     journey.Coordinates.iloc[i-1][0])**2 +
                    (journey.Coordinates.iloc[i+1][1] -
                     journey.Coordinates.iloc[i-1][1])**2)) / 2

            dist_3d = np.sqrt(dist_lateral**2 + elev_change**2)
                # geodesic distance (3d = accounting for elevation), in meters
            Distance[i] = dist_3d
            if Distance[i] != 0 and elev_change != 0:
                Slope[i] = np.arcsin(elev_change / dist_3d)
                    # calculate slope angle in radians from opposite/hypotenuse

    else:
        raise ValueError("Integration method not supported.")

    journey['displacement'] = list(Distance)  # add displacement to dataframe
    journey['slope_rad'] = list(Slope)  # add slope in radians to dataframe
    journey['slope_deg'] = journey['slope_rad'] * 180/np.pi
        # calculate slope angle in degrees


