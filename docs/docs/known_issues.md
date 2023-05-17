---
title: Known Issues
---

There are some known issues with EV-Fleet-Sim that need to be addressed. These include:

* The routing algorithm is negatively effected by inaccurate GPS datapoints. Since GPS modules have an inaccuracy of up to 5\,m, some datapoints in the vehicle's GPS trace may be situated on oncoming lanes. As a result, the routing algorithm would cause the vehicle to perform a U-turn to ensure that the vehicle passes through that waypoint before proceeding with the remaining waypoints along its path. Again, Giliomee et al. [^1] provide a method to compensate for the error.

[^1]: Giliomee et al. (2022) *Simulating Mobility to Plan for Electric Minibus Taxis in Sub-Saharan Africa's Paratransit.* DOI: [doi.org/10.2139/ssrn.4217419](https://doi.org/10.2139/ssrn.4217419). Note: Preprint.
