---
permalink: docs
---

Summary
=======

This program is used to predict the energy usage of a fleet of electric
vehicles. The program receives as input GPS traces of each of the vehicles of
the fleet. These GPS traces can be obtained, for example, by installing
tracking devices onto the vehicles of a fleet for which you want to predict the
electrical energy usage. This is especially useful for projects whereby an
existing petrol/diesel fleet is to be converted to electric vehicles. The
program will analyse the vehicle's driving and stopping patterns in order to
predict the amount of energy used and the amount of time that the vehicle can
be charged during the average day. In addition, the program makes provisions to
calculate how much of the energy can be provided for by renewable-energy
sources.

Please refer to the accompanying open-access journal article publication: [Ray
of hope for sub-Saharan Africa's paratransit: Solar charging of urban electric
minibus taxis in South Africa](https://doi.org/10.1016/j.esd.2021.08.003). The
article shows how this program can be used to derive meaningful results.


Installation
============


To install `EV-Fleet-Sim`, all you have to do is:

1. Install the dependencies listed in the [Installation page]({{site.baseurl}}/docs/installation.html/#dependencies).

2. Run `pip install ev-fleet-sim` in a terminal.


Usage
=====

To run the software, you simply enter `ev-fleet-sim` in a terminal. You can type `ev-fleet-sim --help` to get further usage instructions.

When you run the software, you are presented with a menu which allows you to select which step of the simulation you would like to run. These steps are meant to be run sequentially.

The simulation workflow is described in the [Usage page]({{site.baseurl}}/docs/usage.html).


Contributing
============

If you would like to contribute changes to the software, or even to this documentation, please see the instructions in the [Contributing page]({{site.baseurl}}/docs/contributing.html).


Getting Support
===============

Welcome to our EV-Fleet-Sim community! You can join our community's group chat:
[https://matrix.to/#/#ev-fleet-sim:matrix.org](https://matrix.to/#/#ev-fleet-sim:matrix.org).

For private communcation, you can reach me on my e-mail address: 
`chris <abraham-without-the-A's> [at] gmail [dot] com` or via Matrix:
[https://matrix.to/#/@abrac:matrix.org](https://matrix.to/#/@abrac:matrix.org).

Also, follow the latest news surrounding EV-Fleet-Sim, by following [our RSS feed]({{site.baseurl}}/feed.xml) in your favourite RSS news reader.
